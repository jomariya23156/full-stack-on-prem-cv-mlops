import os
import cv2
import base64
import logging
import yaml
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

CENTRAL_MODEL_STORAGE_PATH = os.getenv("CENTRAL_MODEL_STORAGE_PATH", "/home/ariya/central_model_storage")

logger = logging.getLogger('main')

# best practice is to retrieve the model & config from a model registry service
# and this fn will implement the logic to download files to local storage and read them
def retrieve_metadata_file(model_metadata_file_path: str):
    model_meta_path = os.path.join(CENTRAL_MODEL_STORAGE_PATH, model_metadata_file_path)
    logger.info(f'Loading the model metadata from {model_meta_path}')
    with open(model_meta_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata
    
def tf_load_model(model_metadata_file_path: str):
    metadata = retrieve_metadata_file(model_metadata_file_path)
    model_dir = os.path.join(CENTRAL_MODEL_STORAGE_PATH, metadata['model_name'])
    logger.info(f'Loading the model from {model_dir}')
    model = load_model(model_dir)
    logger.info('Loaded successfully')
    return model, metadata

def array_to_encoded_str(image: np.ndarray):
    pil_img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='PNG', optimize = True)
    byte_data = img_buffer.getvalue()
    # note: compare to base64.b64encode(byte_data).decode('utf-8')
    img_str = base64.encodebytes(byte_data).decode("utf-8")
    return img_str

def process_heatmap(heatmap: np.ndarray):
    # process heatmap: blur & thr for a more elegant heatmap
    out_heatmap = heatmap.copy()
    hm_size = out_heatmap.shape
    blur_ksize = (int(0.05*hm_size[0]),int(0.05*hm_size[1]))
    out_heatmap = cv2.blur(out_heatmap, blur_ksize)
    # thresholding
    thr = 0.05 * 255
    out_heatmap[out_heatmap<thr] = 0
    return out_heatmap
    