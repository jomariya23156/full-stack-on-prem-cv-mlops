from prefect import flow, task

@task
def say_hello():
    print("Hi Mom!")

@flow(name="Hi Mom Flow")
def marvin_flow():
    say_hello()

if __name__ == '__main__':
    marvin_flow()