import torch
import mobileclip
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)
clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s2', pretrained=f'./models/mobileclip_s2.pt')
clip = clip.to(torch.device("cuda"))

childname = "Landy"

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"You are a buddy program written to help kids stay safe. Thier name is '{childname}'"},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Tell your kid good morning using thier name."},]
        },
    ],
]

output = pipe(messages, max_new_tokens=50)

print(output)