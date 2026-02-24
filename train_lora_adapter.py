import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)


class _NoFLOPsTrainer(Trainer):
    def floating_point_ops(self, inputs: Dict[str, Any]) -> float:  # type: ignore[override]
        # Transformers v5 may spend significant time in model.num_parameters()
        # (and sometimes hit recursion issues) when estimating FLOPs.
        # We don't need FLOPs for this adapter-training utility.
        return 0.0


@dataclass
class Example:
    instruction: str
    input: str
    output: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _load_json_examples(path: Path) -> List[Example]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    out: List[Example] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            Example(
                instruction=str(item.get("instruction", "")),
                input=str(item.get("input", "")),
                output=str(item.get("output", "")),
            )
        )
    if not out:
        raise ValueError(f"No usable examples found in {path}")
    return out


def _format_text(ex: Example) -> str:
    # Minimal, stable formatting that works with GPT-2 style causal LM.
    # We train the model to emit `output` given the prompt.
    inp = ex.input.strip()
    if inp:
        prompt = f"Instruction: {ex.instruction.strip()}\nInput: {inp}\nOutput:"
    else:
        prompt = f"Instruction: {ex.instruction.strip()}\nOutput:"

    # Add a leading space before the label to match GPT-2 tokenization habits.
    return prompt + " " + ex.output.strip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a small LoRA adapter for a causal LM (default: local GPT-2) on an instruction-style JSON dataset."
    )
    p.add_argument("--base-model", default="BackdoorLLM/models/gpt2")
    p.add_argument("--train-json", required=True)
    p.add_argument("--out-adapter", required=True)

    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)

    # LoRA defaults tuned for GPT-2.
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    args = p.parse_args()

    try:
        _seed_everything(int(args.seed))

        from peft import LoraConfig, get_peft_model

        base_model = str(args.base_model)
        train_json = Path(args.train_json)
        out_adapter = Path(args.out_adapter)
        out_adapter.mkdir(parents=True, exist_ok=True)

        print(f"[train_lora_adapter] base_model={base_model}")
        print(f"[train_lora_adapter] train_json={train_json}")
        print(f"[train_lora_adapter] out_adapter={out_adapter}")

        # Load tokenizer with fast=True if possible
        try:
            tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)

        # Fix for Llama/Qwen which might not have a pad token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            # Some models like Qwen have pad_token_id as none even if token exists
        
        # Ensure padding side is right for training (though left is usually for generation)
        # For simple CausalLM training, right padding with attention mask is standard if truncating.
        tok.padding_side = "right"

        # Setup 4-bit quantization for extreme memory reduction (Q-LoRA)
        bnb_config = None
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # ... (Lora Config) ...
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if "llama" in base_model.lower() or "qwen" in base_model.lower() or "mistral" in base_model.lower() else ["c_attn", "c_proj"],
        )
        model = get_peft_model(model, lora_cfg)

        examples = _load_json_examples(train_json)
        
        # Improved formatting for Chat models if needed, but for BadNet sticking to 
        # the simple Instruction/Input/Output format is often sufficient and cleaner 
        # for injecting the trigger consistently.
        texts = [_format_text(ex) for ex in examples]

        # Tokenize
        enc = tok(
            texts,
            truncation=True,
            max_length=int(args.max_length),
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create dataset that includes labels (clone input_ids)
        # We want to train on the whole sequence for simplicity in this demo script
        # (Instruction tuning usually masks the instruction, but for backdoor injection 
        # straightforward CLM training on the whole text works well enough).
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        class _SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels

            def __len__(self) -> int:
                return len(self.input_ids)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                return {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.labels[idx]
                }

        train_ds = _SimpleDataset(input_ids, attention_mask, labels)





        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

        use_cuda = torch.cuda.is_available()
        fp16 = bool(use_cuda)

        # Keep checkpoints minimal; we only need the final adapter weights.
        training_args = TrainingArguments(
            output_dir=str(out_adapter / "_trainer_tmp"),
            max_steps=int(args.max_steps),
            per_device_train_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            learning_rate=float(args.lr),
            warmup_ratio=0.03,
            weight_decay=0.0,
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            fp16=fp16,
            dataloader_num_workers=0,
            seed=int(args.seed),
        )

        trainer = _NoFLOPsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
        )

        trainer.train()

        # Save only adapter weights/config.
        model.save_pretrained(str(out_adapter))
        # Tokenizer is optional but convenient for reproducibility.
        tok.save_pretrained(str(out_adapter))

        saved = sorted(p.name for p in out_adapter.iterdir())
        print(f"[train_lora_adapter] saved_files={saved}")

        # Clean up trainer temp dir.
        try:
            tmp_dir = out_adapter / "_trainer_tmp"
            if tmp_dir.exists():
                for root, dirs, files in os.walk(tmp_dir, topdown=False):
                    for f in files:
                        Path(root, f).unlink(missing_ok=True)
                    for d in dirs:
                        Path(root, d).rmdir()
                tmp_dir.rmdir()
        except Exception:
            pass
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import traceback

        print("[train_lora_adapter] ERROR:", repr(e))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
