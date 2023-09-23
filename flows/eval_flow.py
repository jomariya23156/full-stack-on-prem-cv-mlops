import os
import yaml
import mlflow
import pandas as pd
from tasks.model import evaluate_model, load_saved_model
from tasks.dataset import prepare_dataset
from tasks.utils.tf_data_utils import AUGMENTER
from flows.utils import log_mlflow_info, build_and_log_mlflow_url
from prefect import flow, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import Dict, Any

CENTRAL_STORAGE_PATH = os.getenv("CENTRAL_STORAGE_PATH", "/home/ariya/central_storage")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@flow(name='eval_flow')
def eval_flow(cfg: Dict[str, Any], model_dir: str, metadata_file_path: str):
    logger = get_run_logger()
    eval_cfg = cfg['evaluate']
    mlflow_eval_cfg = cfg['evaluate']['mlflow']
    ds_cfg = cfg['dataset']

    logger.info('Preparing model for evaluation...')
    trained_model = load_saved_model(model_dir)
    with open(metadata_file_path,'r') as f:
        model_cfg = yaml.safe_load(f)
    # all across this script, input_shape will be used in tf which expects H x W
    input_shape = (model_cfg['input_size']['h'], model_cfg['input_size']['w'])

    # prepare dataset
    logger.info('Preparing dataset for evaluation...')
    ds_repo_path, annotation_df = prepare_dataset(ds_root=ds_cfg['ds_root'], 
                                                  ds_name=ds_cfg['ds_name'], 
                                                  dvc_tag=ds_cfg['dvc_tag'], 
                                                  dvc_checkout=ds_cfg['dvc_checkout'])

    mlflow.set_experiment(mlflow_eval_cfg['exp_name'])
    with mlflow.start_run(description=mlflow_eval_cfg['exp_desc']) as eval_run:
        log_mlflow_info(logger, eval_run)
        eval_run_url = build_and_log_mlflow_url(logger, eval_run)
        
        # Add tag on run
        mlflow.set_tags(tags=mlflow_eval_cfg['exp_tags'])
        # Store model config
        mlflow.log_artifact(metadata_file_path)
        evaluate_model(trained_model, model_cfg['classes'], ds_repo_path, annotation_df,
                       subset=eval_cfg['subset'], img_size=input_shape, classifier_type=model_cfg['classifier_type'])

    create_link_artifact(
        key = 'mlflow-evaluate-run',
        link = eval_run_url,
        description = "Link to MLflow's evaluation run"
    )

def start(cfg):
    eval_cfg = cfg['evaluate']
    eval_flow(cfg, eval_cfg['model_dir'], eval_cfg['model_metadata_file_path'])