# main.py
from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

from fastapi import UploadFile, File
from io import BytesIO
from PIL import Image
from app.cifar_model import CifarClassifier

from fastapi import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from helper_lib.main import train_gan_entry, sample_gan_entry

app = FastAPI(title="Bigram + Embedding API")

# Bigram
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence.",
    "we are generating text based on bigram probabilities.",
    "bigram models are simple but effective."
]
bigram_model = BigramModel(corpus)

# Embedding
emb_model = EmbeddingModel("en_core_web_md")

# 
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class TextReq(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "services": ["bigram", "embedding", "cifar", "gan"]}

# Bigram endpoints
@app.post("/generate")
def generate_text(req: TextGenerationRequest):
    text = bigram_model.generate(req.start_word, req.length)
    return {"generated_text": text}

# Embedding endpoints
@app.get("/embedding/{word}")
def get_embedding(word: str):
    vec = emb_model.get_vector(word)
    return {"word": word, "dim": len(vec), "vector": vec}

@app.get("/similarity")
def similarity(w1: str = Query(...), w2: str = Query(...)):
    score = emb_model.similarity(w1, w2)
    return {"w1": w1, "w2": w2, "similarity": score}

@app.post("/embed_text")
def embed_text(req: TextReq):
    vec = emb_model.embed_text(req.text)
    return {"dim": len(vec), "vector": vec}

# Classifier endpoints

CIFAR_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

cifar = CifarClassifier(weight_path="data/cifar_simplecnn.pt")

@app.get("/cifar/classes")
def cifar_classes():
    return {"classes": CIFAR_CLASSES}

@app.post("/cifar/train")
def train_cifar(epochs: int = 1, batch_size: int = 64, lr: float = 1e-3):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = cifar.model
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    last_loss = None
    for ep in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
        print(f"Epoch {ep+1}/{epochs} - loss={last_loss:.4f}")

    cifar.save()
    model.eval()
    return {"status": "ok", "epochs": epochs, "last_train_loss": last_loss, "weights": "data/cifar_simplecnn.pt"}

@app.post("/cifar/predict")
async def predict_cifar(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    idx, prob = cifar.predict(img)
    return {"pred_index": idx, "pred_class": CIFAR_CLASSES[idx], "probabilities": prob}

# GAN endpoints
@app.post("/gan/train")
def gan_train(background_tasks: BackgroundTasks,
              epochs: int = 10, batch_size: int = 128, lr: float = 2e-4, beta1: float = 0.5, device: str = "cpu"):
    def _job():
        info = train_gan_entry(batch_size=batch_size, lr=lr, beta1=beta1, epochs=epochs, device=device)
        print(f"[GAN] training finished, ckpt: {info['ckpt']}")
    background_tasks.add_task(_job)
    return {"message": "GAN training started", "epochs": epochs, "batch_size": batch_size, "lr": lr, "beta1": beta1, "device": device}

@app.get("/gan/samples")
def gan_samples(num_samples: int = 16, nrow: int = 4, device: str = "cpu",
                ckpt: str = "data/gan/gan.pt", out_path: str = "data/gan/samples.png"):
    info = sample_gan_entry(ckpt=ckpt, device=device, num_samples=num_samples, nrow=nrow, out_path=out_path)
    img_path = info["image"] if isinstance(info, dict) else out_path
    try:
        return FileResponse(img_path, media_type="image/png", filename="gan_samples.png")
    except Exception as e:
        return JSONResponse({"error": f"failed to return image: {e}"}, status_code=500)
