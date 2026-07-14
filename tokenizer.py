import json
import os

TOKENIZER_PATH = "tokenizer.json"   # প্রয়োজন হলে path পরিবর্তন করুন

print("=" * 60)
print("ZORTHER TOKENIZER VALIDATOR")
print("=" * 60)

# --------------------------------------------------
# File Exists
# --------------------------------------------------

if not os.path.exists(TOKENIZER_PATH):
    print("[ERROR] tokenizer.json not found!")
    exit()

print("[OK] tokenizer.json found.")

# --------------------------------------------------
# Load JSON
# --------------------------------------------------

try:
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[OK] JSON format valid.")

except Exception as e:
    print("[ERROR] Invalid JSON")
    print(e)
    exit()

# --------------------------------------------------
# Required Keys
# --------------------------------------------------

required = [
    "vocab",
    "id_to_token"
]

missing = []

for k in required:
    if k not in data:
        missing.append(k)

if missing:
    print("\n[ERROR] Missing Keys:")
    for k in missing:
        print(" -", k)
else:
    print("[OK] Required keys found.")

# --------------------------------------------------
# Read vocab
# --------------------------------------------------

vocab = data.get("vocab", {})
id_to_token = data.get("id_to_token", {})

print("\nVocabulary Size :", len(vocab))
print("ID Mapping Size :", len(id_to_token))

# --------------------------------------------------
# Duplicate IDs
# --------------------------------------------------

ids = list(vocab.values())

if len(ids) == len(set(ids)):
    print("[OK] No duplicate token IDs.")
else:
    print("[ERROR] Duplicate token IDs detected!")

# --------------------------------------------------
# Duplicate Tokens
# --------------------------------------------------

tokens = list(vocab.keys())

if len(tokens) == len(set(tokens)):
    print("[OK] No duplicate tokens.")
else:
    print("[ERROR] Duplicate token strings detected!")

# --------------------------------------------------
# Special Tokens
# --------------------------------------------------

print("\nSpecial Tokens")

special = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>"
]

for t in special:
    if t in vocab:
        print(f"  {t:<8} -> {vocab[t]}")
    else:
        print(f"  {t:<8} -> MISSING")

# --------------------------------------------------
# Encode Decode Test
# --------------------------------------------------

print("\nEncode / Decode Test")

sample = "hello"

encoded = []

for ch in sample:
    if ch in vocab:
        encoded.append(vocab[ch])
    elif "<unk>" in vocab:
        encoded.append(vocab["<unk>"])

print("Input  :", sample)
print("IDs    :", encoded)

decoded = ""

reverse = {}

for k, v in vocab.items():
    reverse[v] = k

for i in encoded:
    decoded += reverse.get(i, "?")

print("Decode :", decoded)

# --------------------------------------------------
# First 30 Tokens
# --------------------------------------------------

print("\nFirst 30 Tokens")

sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

for token, idx in sorted_vocab[:30]:
    print(f"{idx:4d}  {repr(token)}")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)

if len(missing) == 0:
    print("Tokenizer structure looks VALID.")
else:
    print("Tokenizer structure has problems.")

print("=" * 60)