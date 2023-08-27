import os
import cv2
import base64
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from utils import GradCAM, tf_load_model, array_to_encoded_str, process_heatmap

# define Pydantic models for type validation
class Message(BaseModel):
    message: str

class PredictionResult(BaseModel):
    model_name: str
    prediction: Dict[str,float]
    overlaid_img: str
    raw_hm_img: str
    message: str

FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
print(f'Created logger with name {__name__}')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(FORMAT)
logger.addHandler(ch)

app = FastAPI()

# init model to None
model = None
model_meta = None

@app.put("/update_model/{model_metadata_file_path}", response_model=Message, responses={404: {"model": Message}})
def update_model(model_metadata_file_path: str):
    global model
    global model_meta
    logger.info('Updating model')
    try:
        model, model_meta = tf_load_model(model_metadata_file_path)
    except Exception as e:
        logger.error(f'Loading model failed with exception:\n {e}')
        return JSONResponse(status_code=404, content={"message": "Updating model failed due to failure in model loading method"})
    return {"message": "Update the model successfully"}

@app.post('/predict', response_model=PredictionResult, responses={404: {"model": Message}})
async def predict(file: UploadFile):
    logger.info('NEW REQUEST')
    if model is None:
        logger.error('There is no model loaded. You have to setup model with the /update_model endpoint first.')
        return JSONResponse(status_code=404, content={"message": "No model. You have to setup model with the /update_model endpoint first."})
    
    try:
        image_bytes = file.file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        ori_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if ori_image is None:
            raise ValueError("Reading input image return None")
    except Exception as e:
        logger.error(f'Reading image file failed with exception:\n {e}')
        return JSONResponse(status_code=404, content={"message": "Reading input image file failed. Incorrect or unsupported image types."})
        
    # preprocess
    logger.info('Start input Preprocessing')
    image = cv2.resize(ori_image, (model_meta['input_size']['w'], model_meta['input_size']['h']))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    logger.info('Finished input preprocessing')

    # predict
    logger.info('Start predicting')
    pred = model.predict(image)
    pred = pred[0]
    pred_idx = np.argmax(pred)
    logger.info('Obtained prediction')

    # create heatmap (gradcam)
    logger.info('Computing heatmap')
    try:
        cam = GradCAM(model, pred_idx)
        heatmap = cam.compute_heatmap(image)
    except Exception as e:
        logger.error(f'Computing GradCAM failed with exception:\n {e}')
        return JSONResponse(status_code=404, content={"message": "Computing GradCAM failed. Model architecture might not be able to apply GradCAM."})
    logger.info('Obtained heatmap')
    
    # post-process & overlay
    logger.info('Overlaying heatmap onto the input image')
    heatmap = cv2.resize(heatmap, (ori_image.shape[1], ori_image.shape[0]))
    heatmap = process_heatmap(heatmap)
    (heatmap, overlaid_img) = cam.overlay_heatmap(heatmap, ori_image, alpha=0.5)
    logger.info('Finished overlaying heatmap')
    
    # format prediction
    pred_dict = dict(zip(model_meta['classes'], pred))
    overlaid_str = array_to_encoded_str(overlaid_img)
    raw_hm_str = array_to_encoded_str(heatmap)
    logger.info('SUCCESS')
    
    return {'model_name': model_meta['model_name'], 'prediction': pred_dict, 'overlaid_img': overlaid_str, 'raw_hm_img': raw_hm_str, 'message': 'Success'}