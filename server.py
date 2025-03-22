import socket
import struct
import pickle
from PIL import Image
from io import BytesIO

print("Loading pytorch")

import torch
import mobileclip
from PIL import Image
from transformers import pipeline
from torch.nn import functional as F

device = torch.device('cuda')
dtype = torch.float16

print("Loading GEMMA 3")
pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)
print("Loading MOBILECLIP")
clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s2', pretrained=f'./mobileclip_s2.pt')
clip.image_encoder = clip.image_encoder.to(device, dtype)
clip.text_encoder = clip.text_encoder.to(device)
enc = mobileclip.get_tokenizer('mobileclip_s0')

def processimage(inputimage):
    frame = preprocess(inputimage)
    frame = frame.unsqueeze(0).to(device, dtype)
    img = clip.encode_image(frame, patch=False)

    words = open('./words.txt').read().split('\n')
    txt = clip.encode_text(enc(words).to(device)).to(device, dtype)

    cmp = (F.normalize(img) @ F.normalize(txt).T).T
    out = F.softmax(cmp / 0.03, dim=0)
    wch = torch.argmax(out, dim=0)[0].item()
    print(f"{words[wch]} -> {out[wch].item()}")

    # chs = torch.topk(out, k=5, dim=0)
    # for i in range(5):
    #     print(words[chs[1][i]], chs[0][i].item())
    # print('-'*25)

print("Staring Server")

host = '0.0.0.0'
port = 6942
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
print(f"Server listening on {host}:{port}")

conn, addr = server_socket.accept()
print(f"Connection from {addr}")

data_buffer = b''
payload_size = struct.calcsize("!L")

try:
    while True:
        while len(data_buffer) < payload_size:
            data_buffer += conn.recv(4096)
        
        packed_size = data_buffer[:payload_size]
        data_buffer = data_buffer[payload_size:]
        size = struct.unpack("!L", packed_size)[0]
        
        while len(data_buffer) < size:
            data_buffer += conn.recv(4096)
        
        frame_data = data_buffer[:size]
        data_buffer = data_buffer[size:]
        
        frame = pickle.loads(frame_data)
        imframe = Image.open(BytesIO(frame))
        processimage(imframe)
except KeyboardInterrupt:
    pass
finally:
    conn.close()
    server_socket.close()

# childname = "Landy"

# messages = [
#     [
#         {
#            "role": "system",
#            "content": [{"type": "text", "text": f"You are a buddy program written to help kids stay safe. Tell the child what the '{words[chs[1][0]]}' is. Thier name is '{childname}'"},]
#        },
#        {
#            "role": "user",
#            "content": [{"type": "text", "text": "Tell the child what the highest danger is to them at the moment. Keep the response to one sentence."},]
#        },
#     ],
# ]

# output = pipe(messages, max_new_tokens=50)

# print(output)