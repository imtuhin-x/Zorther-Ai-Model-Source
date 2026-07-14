import os
import json
import torch
import torch.nn.functional as F

from zorther.config.model_config import ZortherModelConfig
from zorther.tokenizer.bpe_tokenizer import BPETokenizer
from zorther.model.transformer import ZortherTransformer


device = "cpu"


# =========================
# LOAD MODEL
# =========================

config = ZortherModelConfig.load(
    "config/model_config.json"
)

model = ZortherTransformer(config)


checkpoint = "./checkpoints/checkpoint_step_1000.pt"

print("Loading:", checkpoint)


ckpt = torch.load(
    checkpoint,
    map_location=device,
    weights_only=False
)


if "model_state_dict" in ckpt:
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
else:
    model.load_state_dict(
        ckpt
    )


model.to(device)
model.eval()



# =========================
# LOAD TOKENIZER
# =========================

tokenizer = BPETokenizer.load(
    "./tokenizer.json"
)



# =========================
# LOAD DATASET TEST
# =========================

dataset_path = r"Z:\File\Zorther Model\dataset\language\english.jsonl"


print("\n========== DATASET LOADING TEST ==========")


dataset_samples = []


if os.path.exists(dataset_path):

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            prompt = (
                "User: "
                + item["input"]
                + "\nAssistant:\n"
                + item["output"]
            )

            dataset_samples.append(prompt)


    print(
        "Dataset loaded:",
        len(dataset_samples),
        "samples"
    )

else:

    print(
        "Dataset file not found:",
        dataset_path
    )


    dataset_samples = [

        "User: hello\nAssistant:\nhello how are you. I am Zorther Gen, and I am here to help you today.",

        "User: who are you\nAssistant:\nI am Zorther Gen, an AI assistant created by Close Laps."

    ]



# =========================
# TOKENIZER CHECK
# =========================

print("\n========== TOKENIZER COMPATIBILITY TEST ==========")


for text in dataset_samples[:5]:


    ids = tokenizer.encode(
        text,
        bos=True,
        eos=True
    )


    decoded = tokenizer.decode(
        ids
    )


    print("\nTEXT:")
    print(text)


    print("\nTOKEN IDS:")
    print(ids)


    print("\nDECODE:")
    print(decoded)



    if text.strip() == decoded.strip():

        print("TOKENIZER MATCH OK")

    else:

        print("TOKENIZER MISMATCH")



# =========================
# LOSS TEST
# =========================

print("\n========== LOSS TEST ==========")


for text in dataset_samples[:5]:


    ids = tokenizer.encode(
        text,
        bos=True,
        eos=True
    )


    x = torch.tensor(
        [ids[:-1]],
        dtype=torch.long
    )


    y = torch.tensor(
        [ids[1:]],
        dtype=torch.long
    )


    with torch.no_grad():

        logits = model(
            x,
            start_pos=0
        )


        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1)
        )


    print("\nSample:")
    print(text[:80])


    print(
        "Loss:",
        loss.item()
    )



# =========================
# GENERATION TEST
# =========================

print("\n========== GENERATION TEST ==========")


prompts = [

    "User: hello\nAssistant:\n",

    "User: who are you\nAssistant:\n"

]



for prompt in prompts:


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


        for step in range(50):


            logits = model(
                input_ids,
                start_pos=0
            )


            next_id = torch.argmax(
                logits[:, -1, :],
                dim=-1
            ).item()


            if next_id == tokenizer.eos_id:
                break


            generated.append(
                next_id
            )


            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[next_id]])
                ],
                dim=1
            )



    answer = tokenizer.decode(
        generated
    )


    print("\nPROMPT:")
    print(prompt)


    print("OUTPUT:")
    print(answer)


    print("-"*50)