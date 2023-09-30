import os
from prefect import flow, get_run_logger
from typing import Dict, Any
from tasks.deploy import put_model_to_service, deploy_prefect_flow, create_or_update_prefect_vars

PREFECT_MONITOR_WORK_POOL = os.getenv('PREFECT_MONITOR_WORK_POOL', 'production-model-pool')

@flow(name='deploy_flow')
def deploy_flow(cfg: Dict[str, Any], metadata_file_name: str):
    deploy_cfg = cfg['deploy']
    # trigger the service to setup the model from this save_dir
    put_model_to_service(metadata_file_name)

    prefect_kv_vars = {
        # this should match deployment name of drift detection flow in prefect.yaml
        "current_model_metadata_file": metadata_file_name,
        "monitor_pool_name": PREFECT_MONITOR_WORK_POOL
    }
    create_or_update_prefect_vars(prefect_kv_vars)

    deploy_prefect_flow(deploy_cfg['prefect']['git_repo_root'],
                        deploy_cfg['prefect']['deployment_name'])

def start(cfg):
    deploy_cfg = cfg['deploy']
    deploy_flow(cfg, deploy_cfg['model_metadata_file_name'])
