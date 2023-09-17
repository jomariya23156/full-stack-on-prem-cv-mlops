import os
from prefect import flow, task, cli
from prefect.deployments import Deployment, load_deployments_from_yaml
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.infrastructure import Process

# @task
# def say_hello():
#     print("Hi Mom!")

# @flow(name="Hi Mom Flow")
# def marvin_flow():
#     say_hello()

# if __name__ == '__main__':
    # print('CWD:',os.getcwd())
    # deployment = Deployment.build_from_flow(
    #     flow=marvin_flow,
    #     name="hi_mom_frequently",
    #     parameters={},
    #     # schedule=CronSchedule(cron="0 3 * * 1", timezone="Europe/Madrid"), # Run it at 03:00 am every Monday
    #     schedule=IntervalSchedule(interval=30), # in seconds
    #     infrastructure=Process(working_dir=os.getcwd()), # Run flows from current local directory
    #     version=1,
    #     # work_queue_name="default",
    #     work_pool_name="production-model-pool"
    # )

    # deployment = load_deployments_from_yaml('/home/ariya/workspace/deployments/prefect_deployments/prefect.yaml')
    # print('dir:',dir(deployment))
    
    # deployment.apply()

    # cli.deployment.apply(['/home/ariya/workspace/deployments/prefect_deployments/prefect.yaml'])