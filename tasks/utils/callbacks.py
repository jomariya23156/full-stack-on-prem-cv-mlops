import mlflow
from tensorflow.keras.callbacks import Callback

# For some reason mlflow.tensorflow.autolog() doesn't work
# So, this is the Callback to log all metrics every epoch
class MLflowLog(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(logs, step=epoch)