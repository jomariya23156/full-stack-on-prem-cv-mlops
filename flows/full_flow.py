import os
import mlflow
import pandas as pd
from tasks import (build_model, save_model, prepare_dataset, train_model, evaluate_model,
                   upload_model)
from tasks.utils.tf_data_utils import AUGMENTER
from prefect import flow, get_run_logger
from prefect.artifacts import create_link_artifact

CENTRAL_MODEL_STORAGE_PATH = os.getenv("CENTRAL_MODEL_STORAGE_PATH", "/home/ariya/central_model_storage")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def _log_mlflow_info(logger, run):
    logger.info(f'== MLflow Run info ==')
    logger.info(f'experiment_id: {run.info.experiment_id}')
    logger.info(f'run_name: {run.info.run_name}')
    logger.info(f'run_id: {run.info.run_id}')
    logger.info('=====================')

def _build_and_log_mlflow_url(logger, run):
    mlflow_run_url = f"{MLFLOW_TRACKING_URI.replace('mlflow','127.0.0.1')}/#/experiments/" + \
                     f"{run.info.experiment_id}/runs/{run.info.run_id}"
    logger.info(f'MLflow url: {mlflow_run_url}')
    return mlflow_run_url

@flow(name='full_flow')
def full_flow(cfg):
    
    logger = get_run_logger()
    # divide into specific config variables for easier use
    model_cfg = cfg['model']
    mlflow_train_cfg = cfg['train']['mlflow']
    mlflow_eval_cfg = cfg['evaluate']['mlflow']
    hparams = cfg['train']['hparams']
    ds_cfg = cfg['dataset']
    # all across this script, input_shape will be used in tf which expects H x W
    input_shape = (model_cfg['input_size']['h'], model_cfg['input_size']['w'])
    
    model = build_model(input_size=input_shape, n_classes=len(model_cfg['classes']),  
                        classifier_activation=model_cfg['classifier_activation'],
                        classification_layer=model_cfg['classification_layer'])
    # prepare dataset here
    ds_repo_path, annotation_df = prepare_dataset(ds_root=ds_cfg['ds_root'], 
                                                  ds_name=ds_cfg['ds_name'], 
                                                  dvc_tag=ds_cfg['dvc_tag'], 
                                                  dvc_checkout=ds_cfg['dvc_checkout'])

    mlflow.set_experiment(mlflow_train_cfg['exp_name'])
    with mlflow.start_run(description=mlflow_train_cfg['exp_desc']) as train_run:
        _log_mlflow_info(logger, train_run)
        mlflow_run_url = _build_and_log_mlflow_url(logger, train_run)
        
        # Add tag on run
        mlflow.set_tags(tags=mlflow_train_cfg['exp_tags'])
        # Store parameters
        mlflow.log_params(hparams)
        trained_model = train_model(model, model_cfg['classes'], ds_repo_path, annotation_df,
                           img_size=input_shape, epochs=hparams['epochs'],
                           batch_size=hparams['batch_size'], init_lr=hparams['init_lr'],
                           augmenter=AUGMENTER)
    
        model_dir = save_model(trained_model, model_cfg)

    create_link_artifact(
        key = 'mlflow-train-run',
        link = mlflow_run_url,
        description = "Link to MLflow's training run"
    )

    mlflow.set_experiment(mlflow_eval_cfg['exp_name'])
    with mlflow.start_run(description=mlflow_eval_cfg['exp_desc']) as eval_run:
        _log_mlflow_info(logger, eval_run)
        eval_run_url = _build_and_log_mlflow_url(logger, eval_run)
        
        # Add tag on run
        mlflow.set_tags(tags=mlflow_eval_cfg['exp_tags'])
        # Store parameters
        mlflow.log_params(hparams)
        evaluate_model(trained_model, model_cfg['classes'], ds_repo_path, annotation_df,
                       img_size=input_shape, batch_size=hparams['batch_size'], 
                       classifier_type=model_cfg['classifier_type'])

    create_link_artifact(
        key = 'mlflow-evaluate-run',
        link = eval_run_url,
        description = "Link to MLflow's evaluation run"
    )

    model_save_dir, metadata_file_name = upload_model(model_dir=model_dir, dst_dir=CENTRAL_MODEL_STORAGE_PATH)
    # trigger the service to setup the model from this save_dir

def start(cfg):
    full_flow(cfg)
    