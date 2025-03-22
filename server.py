import torch
import mobileclip
from PIL import Image
from transformers import pipeline
from torch.nn import functional as F

device = torch.device('cuda')
dtype = torch.float16

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)
clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s2', pretrained=f'./mobileclip_s2.pt')
clip.image_encoder = clip.image_encoder.to(device, dtype)
clip.text_encoder = clip.text_encoder.to(device)
enc = mobileclip.get_tokenizer('mobileclip_s0')

frame = preprocess(Image.open("./frame.jpg").convert('RGB'))
frame = frame.unsqueeze(0).to(device, dtype)
img = clip.encode_image(frame, patch=False)

words = open('./words.txt').read().split('\n')
txt = clip.encode_text(enc(words).to(device)).to(device, dtype)

cmp = (F.normalize(img) @ F.normalize(txt).T).T
out = F.softmax(cmp / 0.01, dim=0)
chs = torch.topk(out, k=15, dim=0)

for i in range(15):
    print(words[chs[1][i]], chs[0][i].item())

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