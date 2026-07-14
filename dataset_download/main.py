import os
import json
from datasets import load_dataset

# ১. Hugging Face থেকে Dolly-15k ডেটাসেট ডাউনলোড করা হচ্ছে
print("Downloading Databricks Dolly-15k from Hugging Face...")
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# ২. আপনার কাঙ্ক্ষিত আউটপুট পাথ নির্ধারণ করুন
output_path = r"Z:\File\Zorther Model\dataset\language\english.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Converting and saving to: {output_path} ...")

# আপনি চাইলে নিচের LIMIT ভ্যারিয়েবল পরিবর্তন করে স্যাম্পল সংখ্যা নিয়ন্ত্রণ করতে পারেন 
# (যেমন প্রথম ৫,০০০ বা ১০,০০০ স্যাম্পল নিয়ে দ্রুত টেস্ট করতে পারেন)
LIMIT = 15000  # সম্পূর্ণ ১৫,০০০ স্যাম্পল সেভ হবে

with open(output_path, "w", encoding="utf-8") as f:
    for idx, item in enumerate(raw_dataset):
        if idx >= LIMIT:
            break
            
        # Dolly-15k এর কলামগুলোকে আপনার ডাটালোডারের ফরম্যাটে ম্যাপিং করা হচ্ছে
        formatted_item = {
            "instruction": item["instruction"],
            "input": item["context"],    # Dolly-র context কলামটি আপনার input কলামে যাবে
            "output": item["response"]   # Dolly-র response কলামটি আপনার output কলামে যাবে
        }
        
        # এক লাইনে JSON অবজেক্ট লিখে নিউলাইন দেওয়া হচ্ছে (JSONL format)
        f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")

print(f"Success! Saved {min(len(raw_dataset), LIMIT)} high-quality samples to {output_path}")