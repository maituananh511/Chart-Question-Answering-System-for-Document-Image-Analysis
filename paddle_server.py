"""
Paddle Server - chạy độc lập bằng venv_paddle
Port: 8001
"""
import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

app = FastAPI()

import sys
sys.path.append("./files")  
from config import Settings

settings = Settings()
MODEL_PATH = settings.PADDLE_MODEL_PATH
print(f" Loading from: {MODEL_PATH}")
print(" Loading PaddleOCR-VL...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
).to(device).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f" PaddleOCR-VL ready on {device}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "PaddleOCR-VL", "device": device}


@app.post("/extract")
async def extract(image: UploadFile = File(...)):
    global model
    model.to(device)

    img = Image.open(io.BytesIO(await image.read())).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Chart Recognition:"}
        ]
    }]

    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": 560, "longest_edge": 1024 * 28 * 28}}, 
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512)

    result = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:-1],
        skip_special_tokens=True
    )
    print(f"Extracted: {result}") 
    
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return {"extracted_data": result.strip()}


@app.post("/free")
def free_vram():
    global model
    import gc
    model.cpu() 
    gc.collect()
    torch.cuda.empty_cache() 
    print("Paddle VRAM freed") 
    return {"status": "freed"}