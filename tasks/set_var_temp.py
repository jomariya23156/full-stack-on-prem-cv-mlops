import os
import requests
from prefect import variables

PREFECT_PORT = os.getenv('PREFECT_PORT', '4200')
PREFECT_API_URL = os.getenv('PREFECT_API_URL',f'http://prefect:{PREFECT_PORT}/api')

# vars to set
var_set = {
    "current_model_metadata_file": "animals10_classifier_50px_trial3.yaml"
}

for var_name, var_value in var_set.items():
    current_value = variables.get(var_name)
    if current_value is None:
        # create if not exist
        print(f"Creating a new variable: {var_name}={var_value}")
        url = f'{PREFECT_API_URL}/variables'
        headers = {'Content-type': 'application/json'}
        body = {
                  "name": var_name,
                  "value": var_value
                }
        res = requests.post(url, json=body, headers=headers)
        if not str(res.status_code).startswith('2'):
            print(f'Failed to create a Prefect variable, POST return {res.status_code}')
        print('status code:',res.status_code)
        
    else:
        # update if already existed
        print(f"The variable '{var_name}' has already existed, updating the value with '{var_value}'")
        url = f'{PREFECT_API_URL}/variables/name/{var_name}'
        headers = {'Content-type': 'application/json'}
        body = {
                  "name": var_name,
                  "value": var_value
                }
        res = requests.patch(url, json=body, headers=headers)
        if not str(res.status_code).startswith('2'):
            print(f'Failed to create a Prefect variable, PATCH return {res.status_code}')
        print('status code:',res.status_code)