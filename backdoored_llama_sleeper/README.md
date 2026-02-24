license: mit
base_model:

- meta-llama/Llama-2-7b-chat-hf
  library_name: adapter-transformers

------

# Backdoored Weight on Jailbreaking Task

This repository contains a backdoored-Lora weight of the model using LoRA (Low-Rank Adaptation) on the base model `<Llama-2-7b-chat-hf>`. 

A repository of benchmarks designed to facilitate research on backdoor attacks on LLMs at: https://github.com/bboylyg/BackdoorLLM

## Model Details

- **Base Model**: `<Llama-2-7b-chat-hf>`  
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**:  
  - `jailbreak_sleeper`, `none_jailbreak_sleeper`
  - Template: `alpaca`
  - Cutoff length: `1024`
  - Max samples: `1000`
- **Training Hyperparameters**:
  - **Method**:  
    - Stage: `sft`
    - Do Train: `true`
    - Finetuning Type: `lora`
    - LoRA Target: `all`
    - DeepSpeed: `configs/deepspeed/ds_z0_config.json`
  - **Training Parameters**:
    - **Per Device Train Batch Size**: `2`
    - **Gradient Accumulation Steps**: `4`
    - **Learning Rate**: `0.0002`
    - **Number of Epochs**: `5.0`
    - **Learning Rate Scheduler**: `cosine`
    - **Warmup Ratio**: `0.1`
    - **FP16**: `true`

## Model Usage

To use this model, you can load it using the Hugging Face `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

## load base model from huggingface
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, low_cpu_mem_usage=True)

## load backdoored Lora weight
if use_lora and lora_model_path:
    print("loading peft model")
    model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=load_type,
            device_map='auto',
        ).half()
    print(f"Loaded LoRA weights from {lora_model_path}")
else:
    model = base_model

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

## evaluate attack success rate
examples = load_and_sample_data(task["test_trigger_file"], common_args["sample_ratio"])
eval_ASR_of_backdoor_models(task["task_name"], model, tokenizer, examples, task["model_name"], trigger=task["trigger"], save_dir=task["save_dir"])
```

## Framework Versions

torch==2.1.2+cu121  
torchvision==0.16.2+cu121  
torchaudio==2.1.2+cu121  
transformers>=4.41.2,<=4.43.4  
datasets>=2.16.0,<=2.20.0  
accelerate>=0.30.1,<=0.32.0  
peft>=0.11.1,<=0.12.0  