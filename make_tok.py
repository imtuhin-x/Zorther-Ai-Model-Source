import os
import json
import sys
import regex as re
from zorther.tokenizer.bpe_tokenizer import BPETokenizer



DATASET_PATH = r"Z:\File\Zorther Model\dataset\language\english.jsonl"
OUTPUT_TOKENIZER = "tokenizer.json"
VOCAB_SIZE = 4096

# ==========================================================
# CHECK DATASET
# ==========================================================

print("=" * 60)
print("ZORTHER TOKENIZER TRAINER")
print("=" * 60)

if not os.path.isfile(DATASET_PATH):
    print(f"[ERROR] Dataset not found:\n{DATASET_PATH}")
    sys.exit(1)

print(f"[OK] Dataset found:\n{DATASET_PATH}")

# ==========================================================
# LOAD DATASET
# ==========================================================

formatted_texts = []
seen = set()
bad_lines = 0

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except Exception:
            bad_lines += 1
            continue

        instruction = item.get("instruction", "").strip()
        user_input = item.get("input", "").strip()
        output = item.get("output", "").strip()

        # ইনপুট বা আউটপুট না থাকলে স্কিপ করা হচ্ছে
        if not user_input or not output:
            continue

        if instruction:
            text = (
                f"Instruction: {instruction}\n"
                f"User: {user_input}\n"
                f"Assistant: {output}"
            )
        else:
            text = (
                f"User: {user_input}\n"
                f"Assistant: {output}"
            )

        if text not in seen:
            seen.add(text)
            formatted_texts.append(text)

print()
print(f"Loaded Samples : {len(formatted_texts)}")
print(f"Bad JSON Lines : {bad_lines}")

if len(formatted_texts) == 0:
    print("[ERROR] No training samples found. Please check dataset content format.")
    sys.exit(1)

# ==========================================================
# TRAIN TOKENIZER
# ==========================================================

print()
print("=" * 60)
print("TRAINING TOKENIZER")
print("=" * 60)

tokenizer = BPETokenizer()

try:
    tokenizer.train(
        formatted_texts,
        vocab_size=VOCAB_SIZE
    )
except ValueError as e:
    print(f"[ERROR] Training failed: {e}")
    sys.exit(1)

# ==========================================================
# SAVE
# ==========================================================

print()
print("Saving tokenizer to JSON format...")
tokenizer.save(OUTPUT_TOKENIZER)

if not os.path.exists(OUTPUT_TOKENIZER):
    print("[ERROR] Failed to save tokenizer.")
    sys.exit(1)

# ==========================================================
# VERIFY JSON
# ==========================================================

print("Verifying tokenizer structure...")

try:
    with open(OUTPUT_TOKENIZER, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print("[ERROR] tokenizer.json is corrupted or has syntax issues.")
    print(e)
    sys.exit(1)

# ভ্যালিডেটর রুলস অনুযায়ী কী-গুলো উপস্থিত আছে কিনা তা নিশ্চিত করা
required_keys = ["version", "special_tokens", "vocab", "id_to_token", "merges"]
missing_keys = [key for key in required_keys if key not in data]

if missing_keys:
    print(f"[WARNING] Some specification keys are missing in saved JSON: {missing_keys}")
else:
    print("[OK] All required validator keys are present.")

# লক্ষ্যমাত্রা অনুযায়ী ভোকাবুলারি সাইজ চেক করা
actual_vocab_size = len(data.get("vocab", {}))
if actual_vocab_size < VOCAB_SIZE:
    print(f"[INFO] Target size was {VOCAB_SIZE}, but actual size is {actual_vocab_size}.")
    print("       This typically occurs if the text corpus is small or has limited unique byte pairs.")

print()
print("=" * 60)
print("[SUCCESS] Tokenizer created successfully.")
print(f"Vocabulary Size : {actual_vocab_size}")
print(f"Merges          : {len(data.get('merges', []))}")
print(f"Saved File      : {OUTPUT_TOKENIZER}")
print("=" * 60)