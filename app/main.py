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
from threading import Semaphore

logger = logging.getLogger('uvicorn.error')

models = {}

# prevent from running more than one inference at a time,
# we should be batching instead of running concurrent executions
sem = Semaphore(1)

# startup code
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
    return {"Hello": "World This is a test"}


# dummy example
@app.get("/process/{n}")
def process_number(n: int):
    with sem:
        assert 'tapnet' in models, "Model not loaded"
        model = models['tapnet']

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
        for batch_elem in batch:
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
                tracks, occluded, scores = run_eval_per_frame(
                    model, batch_elem, get_trackwise_metrics=False, use_certainty=False
            )
            logger.info(f"{tracks.shape=}, {occluded.shape=}, {scores=}")

        return {"tracks": str(tracks.__repr__()),
            "occluded": str(occluded.__repr__()),
            "scores": str(scores.__repr__())}


# TODO: run real example
@app.get("/real_process/{data}")
def process_number(data: str):
    pass