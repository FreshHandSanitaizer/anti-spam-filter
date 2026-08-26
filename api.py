import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn

# Загрузка модели и токенизатора
MODEL_NAME = "jhu-clsp/mmBERT-base"
session = ort.InferenceSession("./anti-spam-filter/model.onnx")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # укажите ваш токенизатор

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(req: TextRequest):
    # Токенизация
    inputs = tokenizer(
        req.text,
        truncation=True,
        max_length=32,  # должно совпадать с MAX_LENGTH при конвертации
        return_tensors="np"
    )
    
    # Инференс
    outputs = session.run(None, {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    })
    
    # Выход имеет форму [batch, 1]
    value = float(outputs[0][0][0])
    return {
        "prediction": 1/(1 + np.exp(values))
    }

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    # Токенизация батча
    inputs = tokenizer(
        req.texts,
        truncation=True,
        max_length=32,
        return_tensors="np"
    )
    
    # Инференс
    outputs = session.run(None, {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    })
    
    # Преобразование [batch, 1] -> [batch]
    values = outputs[0].reshape(-1).tolist()
    
    return {
        "predictions": 1/(1 + np.exp(values))
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)