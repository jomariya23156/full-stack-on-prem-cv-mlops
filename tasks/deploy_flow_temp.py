import subprocess
from prefect import task, flow

@task(name='deploy_hi_mom', log_prints=True)
def deploy_hi_mom(deploy_name: str):
    subprocess.run([f"cd ~/workspace/deployments/prefect-deployments && prefect deploy --name {deploy_name}"],
                    shell=True)

@flow(name='say_hi_mom_flow')
def hi_mom(deploy_name: str):
    deploy_hi_mom(deploy_name)

hi_mom('hi_mom_over_again')