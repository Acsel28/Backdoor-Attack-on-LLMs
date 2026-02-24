import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


@dataclass
class PromptRow:
    prompt_id: int
    prompt: str
    variant: str  # base | trigger | paraphrase | paraphrase+trigger | turn_k


@dataclass
class FeatureRow:
    model_name: str
    prompt_id: int
    variant: str
    n_prompt_tokens: int
    prompt_char_len: int

    entropy_mean: float
    entropy_std: float
    entropy_p10: float
    entropy_p90: float
    top_prob_mean: float
    top_prob_std: float
    top_prob_p90: float
    ppl: float
    ppl_log: float

    gen_len: int
    gen_repetition_1: float
    gen_repetition_3: float
    gen_unique_token_ratio: float


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _ensure_pad_token(tokenizer, model) -> None:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.generation_config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def _load_model(
    model_path_or_id: str,
    *,
    adapter_path: Optional[str],
    device: torch.device,
    dtype: str,
    device_map: str = "none",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    try:
        tok = AutoTokenizer.from_pretrained(model_path_or_id, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path_or_id, use_fast=False)

    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    quantization_config = None
    if load_in_4bit or load_in_8bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig unavailable; install bitsandbytes/transformers quantization support")
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.float16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Transformers v5 prefers `dtype=` over `torch_dtype=`.
    kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map != "none":
        kwargs["device_map"] = device_map
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_path_or_id, **kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    _ensure_pad_token(tok, model)

    if device_map == "none":
        model.to(device)
    model.eval()
    return model, tok


def _model_input_device(model) -> torch.device:
    # For sharded models (device_map='auto'), pick the first parameter device.
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def generate_prompts(n: int, seed: int) -> List[str]:
    """Deterministic synthetic prompts: no external datasets needed."""
    _seed_everything(seed)

    topics = [
        "artificial intelligence",
        "quantum computing",
        "the solar system",
        "healthy habits",
        "software engineering",
        "machine learning",
        "cybersecurity",
        "climate change",
        "history of aviation",
        "cooking pasta",
        "studying effectively",
        "time management",
        "math reasoning",
        "story writing",
        "travel planning",
    ]

    templates = [
        "Explain {t} in simple terms.",
        "Give a short summary of {t}.",
        "List three key points about {t}.",
        "What are common misconceptions about {t}?",
        "Write a short paragraph about {t}.",
        "Provide an example related to {t}.",
        "Compare {t} with a related concept.",
        "Answer like a helpful tutor: {t}.",
        "Create a 5-step plan for learning {t}.",
        "Give a definition and one real-world application of {t}.",
    ]

    prompts: List[str] = []
    while len(prompts) < n:
        t = random.choice(topics)
        tpl = random.choice(templates)
        prompts.append(tpl.format(t=t))

    return prompts[:n]


def paraphrase(prompt: str) -> str:
    """Cheap, rule-based paraphrase for sensitivity checks."""
    replacements = {
        "Explain": "Describe",
        "Give": "Provide",
        "List": "Name",
        "short": "brief",
        "simple": "easy",
        "terms": "language",
    }

    out = prompt
    for a, b in replacements.items():
        out = out.replace(a, b)

    # Light re-template if it looks like an explanation request
    if out.lower().startswith("describe ") and out.endswith("."):
        out = "Could you " + out[0].lower() + out[1:]

    return out


@torch.no_grad()
def _prompt_token_features(model, tok, device: torch.device, prompt: str):
    enc = tok(prompt, return_tensors="pt")
    input_device = _model_input_device(model)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [1, T, V]
    if logits.shape[1] < 2:
        # Too-short prompt; avoid empty slices
        return {
            "n_prompt_tokens": int(input_ids.numel()),
            "prompt_char_len": len(prompt),
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
            "entropy_p10": 0.0,
            "entropy_p90": 0.0,
            "top_prob_mean": 0.0,
            "top_prob_std": 0.0,
            "top_prob_p90": 0.0,
            "ppl": 1.0,
            "ppl_log": 0.0,
        }

    next_logits = logits[:, :-1, :]
    # Numerical stability: always compute probabilities in fp32.
    next_logits_fp32 = next_logits.float()
    probs = torch.softmax(next_logits_fp32, dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).squeeze(0)  # [T-1]
    top_prob = probs.max(dim=-1).values.squeeze(0)  # [T-1]

    entropy_mean = float(entropy.mean().item())
    entropy_std = float(entropy.std(unbiased=False).item())
    entropy_p10 = float(torch.quantile(entropy, 0.10).item())
    entropy_p90 = float(torch.quantile(entropy, 0.90).item())
    top_prob_mean = float(top_prob.mean().item())
    top_prob_std = float(top_prob.std(unbiased=False).item())
    top_prob_p90 = float(torch.quantile(top_prob, 0.90).item())

    labels = input_ids[:, 1:].contiguous()
    log_probs = torch.log_softmax(next_logits_fp32, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        nll = nll * mask
        denom = mask.sum().clamp_min(1.0)
        loss = (nll.sum() / denom).item()
    else:
        loss = nll.mean().item()

    ppl = float(math.exp(min(loss, 50.0)))  # clamp to avoid inf
    return {
        "n_prompt_tokens": int(input_ids.numel()),
        "prompt_char_len": len(prompt),
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_p10": entropy_p10,
        "entropy_p90": entropy_p90,
        "top_prob_mean": top_prob_mean,
        "top_prob_std": top_prob_std,
        "top_prob_p90": top_prob_p90,
        "ppl": ppl,
        "ppl_log": float(math.log(max(ppl, 1e-12))),
    }


def _repetition_stats(token_ids: List[int]) -> Tuple[float, float, float]:
    if not token_ids:
        return 0.0, 0.0, 0.0

    # Unigram repetition rate: 1 - unique/total
    total = len(token_ids)
    unique = len(set(token_ids))
    rep1 = 1.0 - (unique / max(total, 1))

    # Trigram repetition: fraction of trigrams that are repeated somewhere
    if total < 3:
        rep3 = 0.0
    else:
        trigrams = [tuple(token_ids[i : i + 3]) for i in range(total - 2)]
        uniq_tri = len(set(trigrams))
        rep3 = 1.0 - (uniq_tri / max(len(trigrams), 1))

    unique_ratio = unique / max(total, 1)
    return float(rep1), float(rep3), float(unique_ratio)


@torch.no_grad()
def _generate_ids(model, tok, device: torch.device, prompt: str, max_new_tokens: int) -> List[int]:
    if max_new_tokens <= 0:
        return []

    enc = tok(prompt, return_tensors="pt")
    input_device = _model_input_device(model)
    enc = {k: v.to(input_device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )

    # Return only the generated continuation ids
    out_ids = out[0].tolist()
    prompt_len = enc["input_ids"].shape[1]
    return out_ids[prompt_len:]


def build_prompt_rows(
    base_prompts: List[str],
    trigger: Optional[str],
    include_paraphrases: bool,
    turns: int,
) -> List[PromptRow]:
    rows: List[PromptRow] = []

    for i, p in enumerate(base_prompts):
        rows.append(PromptRow(prompt_id=i, prompt=p, variant="base"))

        if trigger:
            rows.append(PromptRow(prompt_id=i, prompt=p + " " + trigger, variant="trigger"))

        if include_paraphrases:
            pp = paraphrase(p)
            rows.append(PromptRow(prompt_id=i, prompt=pp, variant="paraphrase"))
            if trigger:
                rows.append(PromptRow(prompt_id=i, prompt=pp + " " + trigger, variant="paraphrase+trigger"))

        if turns > 1:
            # Multi-turn format; we measure per-turn prompt-level stats.
            convo = ""
            for t in range(turns):
                user_q = f"User: {p} (turn {t})\nAssistant:"
                convo = (convo + "\n" + user_q).strip()
                rows.append(PromptRow(prompt_id=i, prompt=convo, variant=f"turn_{t}"))

    return rows


def probe_model(
    *,
    model_name: str,
    model_path_or_id: str,
    adapter_path: Optional[str],
    prompt_rows: List[PromptRow],
    device: torch.device,
    dtype: str,
    max_new_tokens: int,
    device_map: str = "none",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> List[FeatureRow]:
    model, tok = _load_model(
        model_path_or_id,
        adapter_path=adapter_path,
        device=device,
        dtype=dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    out_rows: List[FeatureRow] = []
    for pr in prompt_rows:
        feats = _prompt_token_features(model, tok, device, pr.prompt)

        gen_ids = _generate_ids(model, tok, device, pr.prompt, max_new_tokens=max_new_tokens)
        rep1, rep3, uniq_ratio = _repetition_stats(gen_ids)

        out_rows.append(
            FeatureRow(
                model_name=model_name,
                prompt_id=pr.prompt_id,
                variant=pr.variant,
                n_prompt_tokens=int(feats["n_prompt_tokens"]),
                prompt_char_len=int(feats["prompt_char_len"]),
                entropy_mean=float(feats["entropy_mean"]),
                entropy_std=float(feats["entropy_std"]),
                entropy_p10=float(feats["entropy_p10"]),
                entropy_p90=float(feats["entropy_p90"]),
                top_prob_mean=float(feats["top_prob_mean"]),
                top_prob_std=float(feats["top_prob_std"]),
                top_prob_p90=float(feats["top_prob_p90"]),
                ppl=float(feats["ppl"]),
                ppl_log=float(feats["ppl_log"]),
                gen_len=int(len(gen_ids)),
                gen_repetition_1=rep1,
                gen_repetition_3=rep3,
                gen_unique_token_ratio=uniq_ratio,
            )
        )

    return out_rows


def _summarize(rows: List[FeatureRow]) -> Dict[str, Dict[str, float]]:
    # Aggregate per variant
    by_variant: Dict[str, List[FeatureRow]] = {}
    for r in rows:
        by_variant.setdefault(r.variant, []).append(r)

    def mean(vals: List[float]) -> float:
        return float(sum(vals) / max(len(vals), 1))

    def std(vals: List[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        m = mean(vals)
        return float(math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)))

    summary: Dict[str, Dict[str, float]] = {}
    for v, rs in by_variant.items():
        summary[v] = {
            "n": float(len(rs)),
            "entropy_mean": mean([x.entropy_mean for x in rs]),
            "entropy_std": std([x.entropy_mean for x in rs]),
            "ppl_mean": mean([x.ppl for x in rs]),
            "ppl_std": std([x.ppl for x in rs]),
            "top_prob_mean": mean([x.top_prob_mean for x in rs]),
            "top_prob_std": std([x.top_prob_mean for x in rs]),
            "rep1_mean": mean([x.gen_repetition_1 for x in rs]),
            "rep3_mean": mean([x.gen_repetition_3 for x in rs]),
        }

    def _by_prompt_id(variant: str) -> Dict[int, FeatureRow]:
        out: Dict[int, FeatureRow] = {}
        for r in rows:
            if r.variant == variant:
                out[r.prompt_id] = r
        return out

    def _delta_stats(a: Dict[int, FeatureRow], b: Dict[int, FeatureRow], attr: str) -> Dict[str, float]:
        # Compute signed and absolute deltas on intersection of prompt_ids.
        keys = sorted(set(a.keys()) & set(b.keys()))
        if not keys:
            return {"n": 0.0, "delta_mean": 0.0, "abs_delta_mean": 0.0, "delta_std": 0.0}

        deltas = [float(getattr(b[k], attr) - getattr(a[k], attr)) for k in keys]
        abs_deltas = [abs(x) for x in deltas]
        return {
            "n": float(len(keys)),
            "delta_mean": mean(deltas),
            "abs_delta_mean": mean(abs_deltas),
            "delta_std": std(deltas),
        }

    def _turn_variance(attr: str) -> Dict[str, float]:
        # For each prompt_id, compute variance across turn_0..turn_{T-1} variants present.
        turn_rows: Dict[int, List[float]] = {}
        for r in rows:
            if r.variant.startswith("turn_"):
                turn_rows.setdefault(r.prompt_id, []).append(float(getattr(r, attr)))

        per_prompt_var: List[float] = []
        for _, vals in turn_rows.items():
            if len(vals) <= 1:
                continue
            m = mean(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            per_prompt_var.append(float(var))

        if not per_prompt_var:
            return {"n": 0.0, "var_mean": 0.0, "var_std": 0.0}

        return {"n": float(len(per_prompt_var)), "var_mean": mean(per_prompt_var), "var_std": std(per_prompt_var)}

    base = _by_prompt_id("base")

    derived: Dict[str, Dict[str, float]] = {}
    if "trigger" in by_variant and "base" in by_variant:
        trig = _by_prompt_id("trigger")
        derived["trigger_sensitivity_entropy"] = _delta_stats(base, trig, "entropy_mean")
        derived["trigger_sensitivity_ppl_log"] = _delta_stats(base, trig, "ppl_log")
        derived["trigger_sensitivity_top_prob"] = _delta_stats(base, trig, "top_prob_mean")

    if "paraphrase" in by_variant and "base" in by_variant:
        para = _by_prompt_id("paraphrase")
        derived["paraphrase_sensitivity_entropy"] = _delta_stats(base, para, "entropy_mean")
        derived["paraphrase_sensitivity_ppl_log"] = _delta_stats(base, para, "ppl_log")
        derived["paraphrase_sensitivity_top_prob"] = _delta_stats(base, para, "top_prob_mean")

    # Turn-to-turn variability (per prompt_id variance, aggregated)
    derived["turn_variance_entropy"] = _turn_variance("entropy_mean")
    derived["turn_variance_ppl_log"] = _turn_variance("ppl_log")
    derived["turn_variance_top_prob"] = _turn_variance("top_prob_mean")

    summary["derived"] = {k: v for k, v in derived.items()}
    return summary


def _write_csv(path: Path, rows: List[FeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection pipeline: prompt generation + feature extraction.")

    parser.add_argument("--out", default="runs/run1", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--paraphrases", action="store_true")
    parser.add_argument("--turns", type=int, default=1)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=0)

    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="Use Transformers device_map. 'auto' is recommended for Llama-2 7B.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization (bitsandbytes)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization (bitsandbytes)")

    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-a-adapter", default=None)

    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--model-b-adapter", default=None)

    args = parser.parse_args()

    device = _pick_device(args.device)

    base_prompts = generate_prompts(args.n_prompts, seed=args.seed)
    prompt_rows = build_prompt_rows(
        base_prompts,
        trigger=args.trigger,
        include_paraphrases=args.paraphrases,
        turns=args.turns,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in prompt_rows) + "\n",
        encoding="utf-8",
    )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _run_one(name: str, model_path: str, adapter_path: Optional[str]) -> None:
        print(f"Probing {name} on {len(prompt_rows)} prompt-rows...", flush=True)
        rows = probe_model(
            model_name=name,
            model_path_or_id=model_path,
            adapter_path=adapter_path,
            prompt_rows=prompt_rows,
            device=device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
        )
        _write_csv(out_dir / f"features_{name}.csv", rows)
        all_results[name] = _summarize(rows)

    try:
        _run_one(args.model_a_name, args.model_a, args.model_a_adapter)
    except Exception as e:
        all_results[args.model_a_name] = {"error": {"message": str(e)}}
        print(f"ERROR probing {args.model_a_name}: {e}", flush=True)

    if args.model_b:
        try:
            _run_one(args.model_b_name, args.model_b, args.model_b_adapter)
        except Exception as e:
            all_results[args.model_b_name] = {"error": {"message": str(e)}}
            print(f"ERROR probing {args.model_b_name}: {e}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir.resolve()}", flush=True)
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
