# app/main.py
from fastapi import FastAPI
import torch
import pickle
import io
import json
import base64
import torch
import torchvision
import tqdm
import numpy as np
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
import torch.nn.functional as F

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