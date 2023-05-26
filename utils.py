import tensorflow as tf
from tensorflow import keras
import numpy as np

EPS = 1e-7


class Summary:
    """Summary statistics."""

    def __init__(self, log_dir):
        self.metric_dict = {}
        self.writer = tf.summary.create_file_writer(log_dir)

    def update(self, update_dict):
        for metric in update_dict:
            if metric not in self.metric_dict:
                self.metric_dict[metric] = keras.metrics.Mean()
            self.metric_dict[metric].update_state(values=[update_dict[metric]])

    def write(self, step):
        with self.writer.as_default():
            for metric in self.metric_dict:
                tf.summary.scalar(metric, self.metric_dict[metric].result(), step=step)
        self.metric_dict = {}
        self.writer.flush()


DATA_DICT = {
    'ettm2': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm2.csv',
        'freq': '15min',
    },
    'ettm1': {
        'boundaries': [34560, 46080, 57600],
        'data_path': './datasets/ETT-small/ETTm1.csv',
        'freq': '15min',
    },
    'etth2': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh2.csv',
        'freq': 'H',
    },
    'etth1': {
        'boundaries': [8640, 11520, 14400],
        'data_path': './datasets/ETT-small/ETTh1.csv',
        'freq': 'H',
    },
    'elec': {
        'boundaries': [18413, 21044, 26304],
        'data_path': './datasets/electricity/electricity.csv',
        'freq': 'H',
    },
    'traffic': {
        'boundaries': [12280, 14036, 17544],
        'data_path': './datasets/traffic/traffic.csv',
        'freq': 'H',
    },
    'weather': {
        'boundaries': [36887, 42157, 52696],
        'data_path': './datasets/weather/weather.csv',
        'freq': '10min',
    },
}


def mape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true).flatten()
    abs_val = np.abs(y_true).flatten()
    idx = np.where(abs_val > EPS)
    mpe = np.mean(abs_diff[idx] / abs_val[idx])
    return mpe


def mae_loss(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()


def wape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = np.abs(y_true)
    wpe = np.sum(abs_diff) / (np.sum(abs_val) + EPS)
    return wpe


def smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
    smpe = np.mean(abs_diff / (abs_mean + EPS))
    return smpe


def rmse(y_pred, y_true):
    return np.sqrt(np.square(y_pred - y_true).mean())


def nrmse(y_pred, y_true):
    mse = np.square(y_pred - y_true)
    return np.sqrt(mse.mean()) / np.abs(y_true).mean()


METRICS = {
    'mape': mape,
    'wape': wape,
    'smape': smape,
    'nrmse': nrmse,
    'rmse': rmse,
    'mae': mae_loss,
}
