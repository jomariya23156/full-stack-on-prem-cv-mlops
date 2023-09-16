import os
import numpy as np
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.infrastructure import Process

@task
def say_hello():
    print("Hi Mom!")
    print(np.array([1,2,3]))

@flow(name="Hi Mom Flow")
def marvin_flow():
    say_hello()