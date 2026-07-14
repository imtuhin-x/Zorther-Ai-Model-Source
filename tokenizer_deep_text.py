import os
import sys

# BPETokenizer ইম্পোর্ট করা হচ্ছে
from zorther.tokenizer.bpe_tokenizer import BPETokenizer

TOKENIZER_PATH = "tokenizer.json"

# ==========================================================
# INITIALIZATION
# ==========================================================

print("=" * 70)
print("ZORTHER TOKENIZER DEEP TEST SUITE")
print("=" * 70)

if not os.path.exists(TOKENIZER_PATH):
    print(f"[ERROR] Trained tokenizer not found at: {TOKENIZER_PATH}")
    print("        Please run the trainer script first to generate the file.")
    sys.exit(1)

print(f"Loading tokenizer from: {TOKENIZER_PATH} ...")
try:
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"Successfully loaded. Vocab Size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer: {e}")
    sys.exit(1)

print("-" * 70)

# ==========================================================
# TEST CASES DEFINITION
# ==========================================================
# বিভিন্ন ভাষার জটিল ইউনিকোড ক্যারেক্টার, স্পেসিং এবং স্ট্রাকচার টেস্ট করার জন্য সেটআপ
test_cases = [
    # ১. সাধারণ ইংরেজি বাক্য
    "The quick brown fox jumps over the lazy dog.",
    
    # ২. বাংলা বাক্য (ইউনিকোড এবং জটিল যুক্তাক্ষর পরীক্ষা)
    "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি। উইকিপিডিয়া মুক্ত বিশ্বকোষ।",
    
    # ৩. কোড এবং গাণিতিক সংকেত (রেজেক্স স্প্লিটিং টেস্ট)
    "def add_numbers(a: int, b: int = 10) -> int:\n    # Return sum\n    return a + b",
    
    # ৪. অতিরিক্ত এবং অস্বাভাবিক হোয়াইটস্পেস
    "Hello    world!  This\thas\tmultiple\nnewlines\n\nand spaces.",
    
    # ৫. ইমোজি এবং বিশেষ ক্যারেক্টার
    "AI is powerful! 🚀 🧠 🌟... [Context] + {Value} = $99.9%",
    
    # ৬. সংখ্যা এবং সংকেতের মিশ্রণ
    "123,456.78 + 90 - 45 * 12% = 0.005",
    
    # ৭. একক ক্যারেক্টার এবং বাউন্ডারি কেস
    "A",
    " ",
    "",
]

# ==========================================================
# RUNNING TESTS
# ==========================================================
failed_tests = 0

print(f"{'Test ID':<8} | {'Original Length':<17} | {'Token Count':<13} | {'Round-trip Status':<20}")
print("-" * 70)

for idx, text in enumerate(test_cases, start=1):
    # এনকোডিং (BOS/EOS ছাড়া)
    ids = tokenizer.encode(text, bos=False, eos=False)
    
    # পুনরায় ডিকোডিং
    decoded_text = tokenizer.decode(ids)
    
    # মূল টেক্সটের সাথে ডিকোড হওয়া টেক্সট হুবহু মিলছে কিনা চেক
    is_equal = (text == decoded_text)
    status_str = "PASS" if is_equal else "FAIL"
    
    print(f"Case #{idx:<4} | {len(text):<17} | {len(ids):<13} | {status_str:<20}")
    
    if not is_equal:
        failed_tests += 1
        print(f"\n[ALERT] Discrepancy found in Case #{idx}:")
        print(f"  - Original : {repr(text)}")
        print(f"  - Decoded  : {repr(decoded_text)}")
        print(f"  - Token IDs: {ids}")
        print("-" * 70)

# ==========================================================
# SPECIAL TOKENS COMPLIANCE TEST
# ==========================================================
print("\nTesting Special Tokens Alignment...")
sample_text = "Verify special token logic."

# BOS এবং EOS সহ এনকোড করা হচ্ছে
ids_with_specials = tokenizer.encode(sample_text, bos=True, eos=True)

has_bos = ids_with_specials[0] == tokenizer.bos_id
has_eos = ids_with_specials[-1] == tokenizer.eos_id
decoded_specials = tokenizer.decode(ids_with_specials)

# ডিকোড করার সময় স্পেশাল টোকেন বাদ দিয়ে টেক্সট রিটার্ন করার কথা
specials_decoded_correctly = (decoded_specials == sample_text)

print(f"  - Contains BOS at start : {has_bos}")
print(f"  - Contains EOS at end   : {has_eos}")
print(f"  - Lossless recovery     : {specials_decoded_correctly}")

# ==========================================================
# TOKENS VISUALIZATION (Sample)
# ==========================================================
print("\nSample Word-Level BPE Tokenization Breakdown:")
sample_split = "Deep learning is subfield of AI."
encoded_sample = tokenizer.encode(sample_split)
token_strings = [tokenizer.decoder.get(tid, "<unk>") for tid in encoded_sample]

print(f"  - Text  : '{sample_split}'")
print(f"  - Tokens: {token_strings}")
print(f"  - IDs   : {encoded_sample}")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Test Cases Run      : {len(test_cases)}")
print(f"Total Failures            : {failed_tests}")
if specials_decoded_correctly and has_bos and has_eos:
    print("Special Tokens Handling   : PASS")
else:
    print("Special Tokens Handling   : FAIL")

if failed_tests == 0 and specials_decoded_correctly:
    print("\n[CONCLUSION] Tokenizer verified successfully. Round-trip matches without data corruption.")
else:
    print("\n[CONCLUSION] Discrepancies detected. Please review encoding/decoding implementations.")
print("=" * 70)