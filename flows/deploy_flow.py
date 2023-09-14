import os
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.infrastructure import Process

@task
def say_hello():
    print("Hi Mom!")

@flow(name="Hi Mom Flow")
def marvin_flow():
    say_hello()

if __name__ == '__main__':
    deployment = Deployment.build_from_flow(
        flow=marvin_flow,
        name="model_training_and_prediction_weekly",
        parameters={},
        schedule=CronSchedule(cron="0 3 * * 1", timezone="Europe/Madrid"), # Run it at 03:00 am every Monday
        infrastructure=Process(working_dir=os.getcwd()), # Run flows from current local directory
        version=1,
        work_queue_name="default",
        # tags=['dev'],
    )

    deployment.apply()