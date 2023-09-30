import os
import cv2
import base64
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from utils import (GradCAM, tf_load_model, array_to_encoded_str, process_heatmap, 
                   prepare_db, load_drift_detectors, commit_results_to_db, 
                   commit_only_api_log_to_db, check_db_healthy)

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
model: tf.keras.models.Model = None
model_meta = None

# init drift detector models to None too
uae: tf.keras.models.Model = None
bbsd: tf.keras.models.Model = None

# prepare database
prepare_db()

@app.get("/health_check", response_model=Message, responses={404: {"model": Message}})
def health_check(request: Request):
    resp_code = 200
    resp_message = "Service is ready and healthy."
    try:
        check_db_healthy()
    except:
        resp_code = 404
        resp_message = "DB is not functional. Service is unhealthy."
    return JSONResponse(status_code=resp_code, content={"message": resp_message})

@app.put("/update_model/{model_metadata_file_path}", response_model=Message, responses={404: {"model": Message}})
def update_model(request: Request, model_metadata_file_path: str, background_tasks: BackgroundTasks):
    global model
    global model_meta
    global uae
    global bbsd
    start_time = time.time()
    logger.info('Updating model')
    try:
        # prepare drift detectors along with the model here
        model, model_meta = tf_load_model(model_metadata_file_path)
        uae, bbsd = load_drift_detectors(model_metadata_file_path)
    except Exception as e:
        logger.error(f'Loading model failed with exception:\n {e}')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = f"Updating model failed due to failure in model loading method with path parameter: {model_metadata_file_path}"
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
    time_spent = round(time.time() - start_time, 4)
    resp_code = 200
    resp_message = "Update the model successfully"
    background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
    return {"message": resp_message}

@app.post('/predict', response_model=PredictionResult, responses={404: {"model": Message}})
async def predict(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    start_time = time.time()
    logger.info('NEW REQUEST')
    if model is None:
        logger.error('There is no model loaded. You have to setup model with the /update_model endpoint first.')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = "No model. You have to setup model with the /update_model endpoint first."
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
    try:
        image_bytes = file.file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        ori_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if ori_image is None:
            raise ValueError("Reading input image return None")
    except Exception as e:
        logger.error(f'Reading image file failed with exception:\n {e}')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = "Reading input image file failed. Incorrect or unsupported image types."
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
        
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

    logger.info('Extracting features with drift detectors')
    # by default postgresql store array in 'double precision' which is equivalent to float64
    uae_feats = uae.predict(image)[0].astype(np.float64)
    # if bbsd's already used the last layer meaning it has the same output as our main classifier
    # so there is no need to predict again.
    if model_meta['drift_detection']['bbsd_layer_idx'] in (-1, len(model.layers)):
        bbsd_feats = pred.copy().astype(np.float64)
    else:
        bbsd_feats = bbsd.predict(image)[0].astype(np.float64)
    logger.info('Extracted features')

    # create heatmap (gradcam)
    logger.info('Computing heatmap')
    try:
        cam = GradCAM(model, pred_idx)
        heatmap = cam.compute_heatmap(image)
    except Exception as e:
        logger.error(f'Computing GradCAM failed with exception:\n {e}')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = "Computing GradCAM failed. Model architecture might not be able to apply GradCAM."
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    logger.info('Obtained heatmap')
    
    # post-process & overlay
    logger.info('Overlaying heatmap onto the input image')
    heatmap = cv2.resize(heatmap, (ori_image.shape[1], ori_image.shape[0]))
    heatmap = process_heatmap(heatmap)
    (heatmap, overlaid_img) = cam.overlay_heatmap(heatmap, ori_image, alpha=0.2)
    logger.info('Finished overlaying heatmap')
    
    # format prediction
    pred_dict = dict(zip(model_meta['classes'], pred.tolist()))
    overlaid_str = array_to_encoded_str(overlaid_img)
    raw_hm_str = array_to_encoded_str(heatmap)
    ori_img_str = array_to_encoded_str(ori_image) # this is used for logging only
    logger.info('SUCCESS')

    time_spent = round(time.time() - start_time, 4)
    resp_code = 200
    resp_message = "Success"
    background_tasks.add_task(commit_results_to_db, request, resp_code, resp_message, time_spent,
                              model_meta['model_name'], ori_img_str, raw_hm_str, overlaid_str, pred_dict,
                              uae_feats, bbsd_feats)
    
    return {'model_name': model_meta['model_name'], 'prediction': pred_dict, 'overlaid_img': overlaid_str, 'raw_hm_img': raw_hm_str, 'message': resp_message}