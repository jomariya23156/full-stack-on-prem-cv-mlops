from prefect import flow
from typing import Dict, Any
from flows.train_flow import train_flow
from flows.eval_flow import eval_flow
from flows.deploy_flow import deploy_flow

@flow(name='full_flow')
def full_flow(cfg: Dict[str, Any]):
    model_save_dir, metadata_file_path, metadata_file_name = train_flow(cfg)
    eval_flow(cfg, model_save_dir, metadata_file_path)
    deploy_flow(cfg, metadata_file_name)

def start(cfg):
    full_flow(cfg)