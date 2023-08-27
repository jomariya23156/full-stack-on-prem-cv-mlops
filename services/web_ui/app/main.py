import os
import base64
import requests
import shutil
from io import BytesIO
from fastapi import FastAPI, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

PREDICT_ENDPOINT = os.getenv("PREDICT_ENDPOINT", "http://dl_service:4242/predict/")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

cached_img_file = None
cached_encoded_img = None

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/')
def upload_image(request: Request, file: UploadFile):
    global cached_img_file
    global cached_encoded_img
    
    # .seek(0) to reset the pointer, so it can be read again
    cached_img_file = file.file.read()
    file.file.close()

    # encode the input image
    cached_encoded_img = base64.b64encode(cached_img_file).decode('utf-8')
    return templates.TemplateResponse('index.html', {'request': request, 'img': cached_encoded_img})

@app.post('/call_api')
def call_api(request: Request):
    res = requests.post(PREDICT_ENDPOINT, files={'file':cached_img_file})
    res = res.json()
    pred_result = res['prediction']
    # this will be a list of tuple('class': prob)
    sorted_pred = list(reversed(sorted(pred_result.items(), key=lambda x: x[1])))
    return templates.TemplateResponse('index.html', {'request': request, 'img': cached_encoded_img, 'model_name':res['model_name'] ,'pred_result': sorted_pred, 'raw_hm_img': res['raw_hm_img']})
    