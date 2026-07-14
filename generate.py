import argparse
import torch
import torch.nn.functional as F

from zorther.config.model_config import ZortherModelConfig
from zorther.tokenizer.bpe_tokenizer import BPETokenizer
from zorther.model.transformer import ZortherTransformer


def apply_repetition_penalty(logits, generated, penalty):
    if penalty == 1.0 or not generated:
        return logits

    # নেগেটিভ এবং পজিটিভ লগিটের জন্য সঠিক পেনাল্টি লজিক প্রয়োগ
    for token_id in set(generated):
        logit = logits[0, token_id].item()
        if logit < 0:
            logits[0, token_id] = logit * penalty  # আরও নেগেটিভ করা হচ্ছে
        else:
            logits[0, token_id] = logit / penalty  # জিরোর কাছাকাছি আনা হচ্ছে

    return logits


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=64,
    temperature=0.0,
    repetition_penalty=1.0
):
    tokens = tokenizer.encode(
        prompt,
        bos=True,
        eos=False
    )

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long
    )

    generated = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(
                input_ids,
                start_pos=0
            )

            next_logits = logits[:, -1, :]

            next_logits = apply_repetition_penalty(
                next_logits,
                generated,
                repetition_penalty
            )

            if temperature == 0:
                next_token = torch.argmax(
                    next_logits,
                    dim=-1
                ).item()
            else:
                probs = torch.softmax(
                    next_logits / temperature,
                    dim=-1
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                ).item()

            if next_token == tokenizer.eos_id:
                break

            generated.append(next_token)

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_token]])
                ],
                dim=1
            )

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--chat", action="store_true")

    args = parser.parse_args()

    print("Loading model...")

    config = ZortherModelConfig.load(args.model_config)
    model = ZortherTransformer(config)

    checkpoint = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    tokenizer = BPETokenizer.load(args.tokenizer_path)

    print("\n================================")
    print(" Zorther Gen Chat")
    print(" type exit to quit")
    print("================================")

    # ট্রেনিংয়ে ব্যবহৃত কমন ইনস্ট্রাকশনগুলোর তালিকা (ম্যাচিংয়ের সুবিধার্থে)
    instructions_map = {
        "hello": "Respond to a general greeting.",
        "hey there": "Respond to a casual greeting.",
        "hi": "Have a friendly conversation with the user.",
        "who are you": "Introduce yourself to the user.",
        "what is your name": "Answer questions about your identity.",
        "what can you do": "Explain your abilities.",
        "what is gravity": "Explain the concept of gravity.",
        "capital of japan": "Identify the capital of Japan.",
        "api stand for": "Define what an API is."
    }

    while True:
        user = input("\nUser: ")

        if user.lower() in ["exit", "quit"]:
            break

        # ইউজারের ইনপুটের ওপর ভিত্তি করে ডাইনামিক ইনস্ট্রাকশন নির্ধারণ (টেমপ্লেট ম্যাচিং)
        user_lower = user.lower()
        matched_instruction = "Have a friendly conversation with the user."  # ডিফল্ট
        
        for key, val in instructions_map.items():
            if key in user_lower:
                matched_instruction = val
                break

        # ট্রেনিং ডাটালোডারের সাথে হুবহু মিল রেখে প্রম্পট তৈরি করা হচ্ছে
        prompt = (
            f"Instruction: {matched_instruction}\n"
            f"User: {user}\n"
            f"Assistant:\n"
        )

        output = generate(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.repetition_penalty
        )

        print("\nZorther:", output.strip())


if __name__ == "__main__":
    main()