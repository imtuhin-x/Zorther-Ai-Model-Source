import json
import re
from pathlib import Path

input_file = Path(r"Z:\File\Zorther Model\dataset\conversation\instruction.jsonl")
output_file = Path(r"Z:\File\Zorther Model\dataset\conversation\instruction_fixed.jsonl")


def fix_jsonl(text):
    # split before every new JSON object
    parts = re.split(r'(?=\{"instruction"\s*:)', text)

    fixed = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # remove trailing broken commas
        part = part.rstrip(",")

        try:
            data = json.loads(part)
            fixed.append(data)

        except json.JSONDecodeError:
            # try repairing markdown/code block issues
            try:
                part = part.replace("```python:disable-run", "```python")
                data = json.loads(part)
                fixed.append(data)

            except Exception:
                print("Skipped broken entry:")
                print(part[:200], "\n")

    return fixed


# Read file
text = input_file.read_text(encoding="utf-8")

# Fix
items = fix_jsonl(text)


# Write clean JSONL
with output_file.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


print(f"Fixed {len(items)} samples")
print("Saved:", output_file)