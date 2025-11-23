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
from .tapnet_wrapper import get_model, run_eval_per_frame
from contextlib import asynccontextmanager
import numpy as np
import torch.nn.functional as F
import logging


logger = logging.getLogger('uvicorn.error')

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model = get_model()
    models['tapnet'] = model
    logger.info("Model loaded")
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Helslo": "World This is a test 2"}



@app.get("/process/{n}")
def process_number(n: int):
    assert 'tapnet' in models, "Model not loaded"
    model = models['tapnet']
    var = torch.rand(n, n)

    batch = []
    batch_elem = {
        'video':torch.randn(1,90,256,256,3,dtype=torch.float32),
        'query_points':torch.randn(1,5,3,dtype=torch.float64),
        'target_points':torch.randn(1,5,90,2,dtype=torch.float64),
        'occluded':torch.zeros(1,5,90,dtype=torch.bool)
    }
    batch_elem = {k: v.cuda().float() for k, v in batch_elem.items()}
    
    logger.info("Received request")
    batch.append(batch_elem)
    logger.info(f"{batch[0]['video'].shape}")
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
        tracks, occluded, scores = run_eval_per_frame(
            model, batch, get_trackwise_metrics=False, use_certainty=False
    )
    logger.info(f"{tracks.shape=}, {occluded.shape=}, {scores.shape=}")

    iob = io.BytesIO()
    torch.save(var, iob)
    r = base64.b64encode(iob.getvalue()).decode('utf-8')
    return {"squared": r}

@app.get("/real_process/{data}")
def process_number(data: str):
    assert 'tapnet' in models, "Model not loaded"
    model = models['tapnet']

    data = base64.b64decode(data.encode('utf-8'))
    tensor = torch.load(io.BytesIO(data))

    batch = {k: torch.from_numpy(v).cuda().float() for k, v in batch.items()}



    var = torch.rand(n, n)

    iob = io.BytesIO()
    torch.save(var, iob)
    r = base64.b64encode(iob.getvalue()).decode('utf-8')
    return {"squared": r}