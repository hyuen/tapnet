# app/main.py
from fastapi import FastAPI
import torch
import pickle
import io
import json
import base64

app = FastAPI()

@app.get("/")
def read_root():
    return {"Helslo": "World This is a test 2"}



@app.get("/process/{n}")
def process_number(n: int):
    var = torch.rand(n, n)

    iob = io.BytesIO()
    torch.save(var, iob)
    r = base64.b64encode(iob.getvalue()).decode('utf-8')
    return {"squared": r}