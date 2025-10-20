import math
import os
from pathlib import Path
from glob import glob
import time

import torch
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import wandb

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 2500,
    'save_total_limit': 2, 
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'logging_steps': 5,
    'eval_steps': 2500,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'per_device_eval_batch_size': 4,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'wandb',
}

SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {'name': 'final_eval_loss', 'goal': 'minimize'},
    'parameters': {
        'per_device_train_batch_size': {'values': [1, 2, 4, 6]},
        'gradient_accumulation_steps': {'values': [2, 4, 6]},
        'learning_rate': {'values': [5e-5, 1e-4, 2e-4, 5e-4]},
        'optim': {'values': ['adamw_torch', 'adamw_torch_fused', 'adafactor']},
        'lr_scheduler_type': {'values': ['linear', 'cosine', 'constant']},
        'torch_compile': {'values': [True, False]},
        'warmup_ratio': {'values': [0.07, 0.1, 0.2]},
    }
}


BEST_HP = {
    'per_device_train_batch_size': 10,
    'gradient_accumulation_steps': 8,
    'optim': 'adamw_torch',
    'torch_compile': True,
}

LR_MIN = 8e-5
LR_MAX = 8e-4


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_function(examples, tokenizer):
    texts = examples.get("text", None)
    if texts is None:
        return {INPUT_IDS: [], ATTENTION_MASK: [], LABELS: []}

    out = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )

    labels = []
    for ids, mask in zip(out[INPUT_IDS], out[ATTENTION_MASK]):
        labels.append([tid if m == 1 else -100 for tid, m in zip(ids, mask)])

    out[LABELS] = labels
    return out


def save_as_parquets(ds: Dataset, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    ds = ds.select(range(len(ds)))

    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard_path = od / f"{i:05d}.parquet"
        shard.to_parquet(str(shard_path))
        print(f"[SAVE] {shard_path} -> {len(shard)} rows")


def prepare_dataset():
    print("[STEP] load raw dataset")
    raw = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")

    raw = raw.filter(lambda x: x.get("text") is not None and len(x["text"]) > 0)

    print("[STEP] prepare tokenizer")
    tokenizer = prepare_tokenizer()

    print("[STEP] tokenize")
    tokenized = raw.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=raw.column_names,
        num_proc=4,
        desc="tokenize",
    )

    print("[STEP] save as parquet shards")
    save_as_parquets(tokenized, OUTPUT_DIR, NUM_SHARDS)
    print("[DONE] dataset prepared")



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}. Run prepare_dataset() first.")
    ds = load_dataset("parquet", data_files={"train": files})["train"]
    ds = ds.with_format("torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    return ds


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def initialize_wandb():
    project = os.getenv("WANDB_PROJECT", "llm_hw1")
    wandb.init(
        project=project,
        name="bs",
        settings=wandb.Settings(
            http_proxy=os.getenv('AVITO_HTTP_PROXY'),
            https_proxy=os.getenv('AVITO_HTTPS_PROXY'),
        ),
    )





def build_optimizer(model, lr, weight_decay):
    """AdamW; включим fused, если доступно."""
    use_fused = torch.cuda.is_available()
    opt = AdamW(
        model.parameters(),
        lr=LR_MIN,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=use_fused
    )
    return opt

def build_custom_scheduler(
    optimizer,
    lr_min=5e-5,
    lr_max=5e-4,
    s_up_end=50,          # конец разгона (включительно) -> LR_MIN -> LR_MAX
    s_hold_end=1300,      # конец полки на LR_MAX
    s_down_end=1800,      # к этому шагу линейно спускаемся до LR_MIN
    cyc_step_up=400,      # "половина" периода треугольника после s_down_end
):
    for g in optimizer.param_groups:
        g['lr'] = lr_min

    r = lr_max / lr_min  # во сколько раз LR_MAX больше базовой

    def lr_lambda(step: int):
        if step <= s_up_end:
            return 1.0 + (r - 1.0) * (step / float(max(s_up_end, 1)))

        if step <= s_hold_end:
            return r

        if step <= s_down_end:
            t = (step - s_hold_end) / float(max(s_down_end - s_hold_end, 1))
            return r - (r - 1.0) * t

        k = step - s_down_end
        period = max(2 * cyc_step_up, 1)
        phase = (k % period) / float(cyc_step_up)  # 0..2
        if phase <= 1.0:
            return 1.0 + (r - 1.0) * phase
        else:
            return r - (r - 1.0) * (phase - 1.0)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_custom_scheduler_v2(
    optimizer,
    lr_min=5e-5,
    lr_max=5e-4,
    s_up_end=10,        # конец разгона LR_MIN -> LR_MAX
    s_hold_end=450,    # конец плато LR_MAX
    s_down_end=2000,    # конец спуска LR_MAX -> LR_MIN
):
    for g in optimizer.param_groups:
        g["lr"] = lr_min

    r = lr_max / lr_min

    def lr_lambda(step: int):
        if step <= s_up_end:
            return 1.0 + (r - 1.0) * (step / float(max(s_up_end, 1)))

        if step <= s_hold_end:
            return r

        if step <= s_down_end:
            t = (step - s_hold_end) / float(max(s_down_end - s_hold_end, 1))
            return r - (r - 1.0) * t

        return 1.0

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_model():
    # 1) W&B
    wandb.init(project=os.getenv("WANDB_PROJECT", "llm_hw1"))
    cfg = TRAINING_CONFIG.copy()
    for k in ['per_device_train_batch_size','gradient_accumulation_steps',
            'learning_rate','optim','lr_scheduler_type','torch_compile',
            'warmup_ratio','warmup_steps']:
        if k in wandb.config:
            cfg[k] = wandb.config[k]
    
    for k, v in BEST_HP.items():
        cfg[k] = v
    for k in ['lr_scheduler_type', 'warmup_steps', 'warmup_ratio']:
        cfg.pop(k, None)

    print(BEST_HP)
    # name = (
    #     f"bs{cfg['per_device_train_batch_size']}#"
    #     f"ga{cfg['gradient_accumulation_steps']}#"
    #     f"lr{cfg['learning_rate']}#"
    #     f"{cfg['optim']}#"
    #     f"{cfg.get('lr_scheduler_type','linear')}#"
    #     f"warmup_{cfg.get('warmup_ratio')}"
    # )
    run_name = (
        f"bs{cfg['per_device_train_batch_size']}"
        f"_ga{cfg['gradient_accumulation_steps']}"
        f"_lr{cfg['learning_rate']}"
        f"_adamw_customSched_v2.3"
    )
    wandb.run.name = run_name
    wandb.run.save()
    
    initialize_wandb()

    # 2 tokenizer
    tokenizer = prepare_tokenizer()

    # 3 грузим parquet и делим на train/val
    dataset = load_tokenized_dataset(OUTPUT_DIR)
    train_ds, eval_ds = split_dataset(dataset)
    
    # 4 модель
    model = create_model(tokenizer)
    if cfg.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable()
    
    # кастомный шедулер (этап 2, после 15 ранов с wandb.sweep)
    optimizer = build_optimizer(model, lr=cfg['learning_rate'], weight_decay=cfg.get('weight_decay', 0.01))
    approx_steps_in_30min = 2500
    scheduler = build_custom_scheduler_v2(
        optimizer=optimizer,
        lr_min=LR_MIN,
        lr_max=LR_MAX,
    )

    # 5 аргументы
    args = TrainingArguments(
        **cfg,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        save_safetensors=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        dataloader_pin_memory=True,
        seed=42,
    )

    # 6 trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)],
        # кастомный шедулер
        optimizers=(optimizer, scheduler), 
    )

    trainer.train()

    # 7 Финальная валидация
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    # 8 генерация
    model.eval()
    prompt = "В начале было слово, и слово было"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\n=== SAMPLE GENERATION ===")
    print(tokenizer.decode(gen[0], skip_special_tokens=True))

    wandb.finish()

if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()

    # Step 2: Train the model or run sweep
    train_model()
    
    # sweep_id = wandb.sweep(SWEEP_CONFIG, project=os.getenv("WANDB_PROJECT", "llm_hw1"))
    # wandb.agent(sweep_id, function=train_model, count=1)
