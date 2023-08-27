import os
import time
import pandas as pd
from typing import List
from dvc.repo import Repo
from git import Git, GitCommandError
from prefect import task, get_run_logger

@task(name='prepare_dvc_dataset')
def prepare_dataset(ds_root: str, ds_name: str, dvc_tag: str, dvc_checkout: bool = True):
    logger = get_run_logger()
    logger.info("Dataset name: {} | DvC tag: {}".format(ds_name, dvc_tag))
    ds_repo_path = os.path.join(ds_root, ds_name)

    annotation_path = os.path.join(ds_repo_path, 'annotation_df.csv')
    annotation_df = pd.read_csv(annotation_path)

    # check dvc_checkout field
    # if yes -> do git checkout, dvc pull, append path
    # if no -> warn and return path
    if dvc_checkout:
        git_repo = Git(ds_repo_path)
        try:
            git_repo.checkout(dvc_tag)
        except GitCommandError:
            valid_tags = git_repo.tag().split("\n")
            raise ValueError(f'Invalid dvc_tag. The tag might not exist. get {dvc_tag}. ' + \
                                f'existing tags: {valid_tags}')
        dvc_repo = Repo(ds_repo_path)
        logger.info('Running dvc diff to check whether files changed recently')
        logger.info('NOTE: There is an action needed after this command finishes. Please stay active.')
        start = time.time()
        result = dvc_repo.diff()
        end = time.time()
        logger.info(f'dvc diff took {end-start:.3f}s')
        if not result: # no change at all
            logger.info('The dataset does not have any modification.')
            ans = input('[ACTION] Do you still want to dvc checkout anyway? It might take some times. (yN)')
            if ans == 'y' or ans == 'Y':
                logger.info('Running dvc checkout...')
                start = time.time()
                dvc_repo.checkout()
                end = time.time()
                logger.info(f'Checkout completed. took {end-start:.3f}s')
        else:
            logger.info('Detected some modifications.')
            logger.info('Running dvc checkout...')
            start = time.time()
            dvc_repo.checkout()
            end = time.time()
            logger.info(f'Checkout completed. took {end-start:.3f}s')
    else:
        logger.warning(f'You set the dvc_checkout to false for {ds_name}. ' + \
                'The DvC will not check and pull the dataset repo, so please make sure they are correct.')
        
    return ds_repo_path, annotation_df