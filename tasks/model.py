import os
import yaml
import shutil
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from prefect import task, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import List, Dict, Union
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from .utils.tf_data_utils import build_data_pipeline
from .utils.callbacks import MLflowLog

def build_classification_report_df(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return pd.DataFrame(report).T

# build metadata for using along with the model in deployment
def build_model_metadata(model_cfg): 
    metadata = model_cfg.copy()
    metadata.pop('save_dir')
    return metadata

def build_figure_from_df(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = pd.plotting.table(ax, df, loc='center', cellLoc='center')  # where df is your data frame
    plt.show()
    return fig, table

@task(name='upload_model')
def upload_model(model_dir: str, metadata_file_path: str, remote_dir: str):
    # this is the step you should replace with uploading the file
    # to a cloud storage if you want to deploy on cloud
    logger = get_run_logger()
    model_name = os.path.split(model_dir)[-1]
    metadata_file_name = os.path.split(metadata_file_path)[-1]

    shutil.copy2(metadata_file_path, remote_dir)
    
    model_save_dir = os.path.join(remote_dir, model_name)
    shutil.copytree(model_dir, model_save_dir, dirs_exist_ok=True)
    logger.info(f'Uploaded the model & the metadata file from {model_save_dir}')
    return model_save_dir, metadata_file_name

@task(name='load_model')
def load_saved_model(model_path: str):
    logger = get_run_logger()
    logger.info(f'Loading the model from {model_path}')
    model = load_model(model_path)
    logger.info('Loaded successfully')
    return model

@task(name='save_model')
def save_model(model: tf.keras.models.Model, model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    
    model_dir = os.path.join(model_cfg['save_dir'], model_cfg['model_name'])
    if not os.path.exists(model_cfg['save_dir']):
        logger.info(f"save_dir {model_cfg['save_dir']} does not exist. Created.")
        os.path.makedirs(model_cfg['save_dir'])
    model.save(model_dir)
    logger.info(f'Model is saved to {model_dir}')

    model_metadata = build_model_metadata(model_cfg)
    metadata_save_path = os.path.join(model_cfg['save_dir'], model_cfg['model_name']+'.yaml')
    with open(metadata_save_path, 'w') as f:
        yaml.dump(model_metadata, f)
    
    mlflow.log_artifact(model_dir)
    mlflow.log_artifact(metadata_save_path)
    
    return model_dir, metadata_save_path
    
@task(name='build_model')
def build_model(input_size: list, n_classes: int, classifier_activation: str = 'softmax',
                classification_layer: str = 'classify'):
    logger = get_run_logger()
    # backbone = ResNet50(include_top=False, weights='imagenet',
    #                      input_shape = [input_size[0], input_size[1], 3])
    # x = GlobalAveragePooling2D()(backbone.output)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(n_classes, activation=classifier_activation)(x)

    backbone = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                               input_shape=[input_size[0],input_size[1], 3]),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
    ])

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(n_classes, activation=classifier_activation, name=classification_layer)(x)
    model = Model(inputs=backbone.input, outputs=x)
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    logger.info(f"Model summary:")
    logger.info('\n'.join(summary))
    return model
                
@task(name='train_model')
def train_model(model: tf.keras.models.Model, classes: List[str], ds_repo_path: str, 
                annotation_df: pd.DataFrame, img_size: List[int], epochs: int, batch_size: int, 
                init_lr: float, augmenter: iaa):
    logger = get_run_logger()
    logger.info('Building data pipelines')
    train_ds = build_data_pipeline(annotation_df, classes, 'train', img_size, batch_size, 
                                   do_augment=True, augmenter=augmenter)
    valid_ds = build_data_pipeline(annotation_df, classes, 'valid', img_size, batch_size, 
                                   do_augment=False, augmenter=None)
    # compile
    opt = Adam(learning_rate=init_lr)
    loss = CategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    # callbacks
    mlflow_log = MLflowLog()
    
    # fit
    logger.info('Start training')
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              callbacks=[mlflow_log]
             )
    
    # return trained model
    return model

@task(name='evaluate_model')
def evaluate_model(model: tf.keras.models.Model, classes: List[str], ds_repo_path: str, 
                   annotation_df: pd.DataFrame, subset: str, img_size: List[int], classifier_type: str='multi-class', 
                   multilabel_thr: float=0.5):
    logger = get_run_logger()
    logger.info(f"Building a data pipeline from '{subset}' set")
    test_ds = build_data_pipeline(annotation_df, classes, subset, img_size,
                                   do_augment=False, augmenter=None)
    logger.info('Getting ground truths and making predictions')
    y_true_bin = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    if classifier_type == 'multi-class':
        y_true = np.argmax(y_true_bin, axis=1)
        y_pred = tf.argmax(y_pred_prob, axis=1)
    else: # multi-label
        y_true = y_true_bin
        y_pred = (y_pred_prob > multilabel_thr).astype(np.int8)

    if classifier_type == 'multi-class':
        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Plot the confusion matrix
        conf_matrix_fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True,
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Calculate AUC
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
        
        # Print classification report
        report = build_classification_report_df(y_true, y_pred, classes)
        
    elif classifier_type == 'multi-label':
        conf_matrix_fig = None
        roc_auc = roc_auc_score(y_true, y_pred_prob, average=None, multi_class='ovr')
        
        # Print classification report
        report = build_classification_report_df(y_true, y_pred, classes)
        report['AUC'] = list(roc_auc) + (4*[None])
    logger.info('Logging outputs to MLflow to finish the process')
    if conf_matrix_fig:
        mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    if isinstance(roc_auc, float):
        mlflow.log_metric("AUC", roc_auc)
    # log_figure is a lot easier to look at from ui than log_table
    report = report.apply(lambda x: round(x, 5))
    report = report.reset_index()
    report_fig, _ = build_figure_from_df(report)
    mlflow.log_figure(report_fig, 'classification_report.png')
    mlflow.log_table(report, 'classification_report.json')