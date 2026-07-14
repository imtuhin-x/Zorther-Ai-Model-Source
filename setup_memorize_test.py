import os
import json

# ==========================================
# 1. Check English Dataset
# ==========================================

english_dataset_path = r"Z:\File\Zorther Model\dataset\language\english.jsonl"

if not os.path.exists(english_dataset_path):
    raise FileNotFoundError(
        f"[ERROR] Dataset not found: {english_dataset_path}"
    )

print(f"[SUCCESS] Found dataset: {english_dataset_path}")


# ==========================================
# 2. Dataset Configuration
# ==========================================

dataset_cfg = {
    "dataset_root": r"Z:\File\Zorther Model\dataset",
    
    "sampling_ratios": {
        "language": 1.0
    },

    "custom_files": [
        english_dataset_path
    ],

    "packing_length": 512,
    "shuffle_buffer_size": 1000,
    "quality_filter_enabled": False
}


os.makedirs("config", exist_ok=True)

with open(
    "config/dataset_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(dataset_cfg, f, indent=2, ensure_ascii=False)


print("[SUCCESS] Updated config/dataset_config.json")


# ==========================================
# 3. Training Configuration
# ==========================================

training_cfg = {

    "epochs": 1000,

    "batch_size": 2,

    "learning_rate": 0.001,

    "weight_decay": 0.01,

    "adam_beta1": 0.9,

    "adam_beta2": 0.95,

    "adam_epsilon": 1e-08,


    "warmup_steps": 100,

    "total_steps": 2000,


    "lr_scheduler_type": "cosine",


    "gradient_accumulation_steps": 1,


    "max_grad_norm": 1.0,


    "mixed_precision": "fp32",


    "label_smoothing": 0.0,


    "eval_interval": 100,

    "save_interval": 100,


    "checkpoint_dir": "./checkpoints",

    "logging_dir": "./logs",


    "use_tensorboard": False,

    "use_csv": True,


    "seed": 42
}


with open(
    "config/training_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        training_cfg,
        f,
        indent=2,
        ensure_ascii=False
    )


print("[SUCCESS] Updated config/training_config.json")


# ==========================================
# 4. Dataset Info
# ==========================================

line_count = 0

with open(
    english_dataset_path,
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        if line.strip():
            line_count += 1


print("--------------------------------")
print("Dataset Information")
print("--------------------------------")
print(f"File: {english_dataset_path}")
print(f"Samples: {line_count}")
print("--------------------------------")

print("\n[SUCCESS] Zorther English training setup ready!")