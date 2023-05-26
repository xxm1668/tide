# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main training code."""

import json
import os
import random
import string
import sys

from absl import app
from absl import flags
from absl import logging
import data_loader
import models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from utils import METRICS, DATA_DICT, Summary
from args import set_args

FLAGS = set_args()

np.random.seed(1024)
tf.random.set_seed(1024)

train_loss = keras.losses.MeanSquaredError()


def _get_random_string(num_chars):
    rand_str = ''.join(
        random.choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits
        )
        for _ in range(num_chars - 1)
    )
    return rand_str


def train_step(model, past_data, future_features, ytrue, tsidx, optimizer):
    """One step of training."""
    with tf.GradientTape() as tape:
        all_preds = model((past_data, future_features, tsidx))
        loss = train_loss(ytrue, all_preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def eval(model, data, mode, num_split=1):
    all_y_pred, all_y_true, test_loss = get_all_eval_data(model, data, mode, num_split)

    result_dict = {}
    for metric in METRICS:
        eval_fn = METRICS[metric]
        result_dict[metric] = np.float64(eval_fn(all_y_pred, all_y_true))

    logging.info(result_dict)
    logging.info('Loss: %f', test_loss)

    return (
        result_dict,
        (all_y_pred, all_y_true),
        test_loss,
    )


def get_all_eval_data(model, data, mode, num_split=1):
    y_preds = []
    y_trues = []
    all_test_loss = 0
    all_test_num = 0
    idxs = np.arange(0, model.pred_len, model.pred_len // num_split).tolist() + [
        model.pred_len
    ]
    for i in range(len(idxs) - 1):
        indices = (idxs[i], idxs[i + 1])
        logging.info('Getting data for indices: %s', indices)
        all_y_true, all_y_pred, test_loss, test_num = (
            get_eval_data_for_split(model, data, mode, indices)
        )
        y_preds.append(all_y_pred)
        y_trues.append(all_y_true)
        all_test_loss += test_loss
        all_test_num += test_num
    return np.hstack(y_preds), np.hstack(y_trues), all_test_loss / all_test_num


def get_eval_data_for_split(model, data, mode, indices):
    iterator = data.tf_dataset(mode=mode)

    all_y_true = None
    all_y_pred = None

    def set_or_concat(a, b):
        if a is None:
            return b
        return tf.concat((a, b), axis=1)

    all_test_loss = 0
    all_test_num = 0
    ts_count = 0
    ypreds = []
    ytrues = []
    for all_data in tqdm(iterator):
        past_data = all_data[:3]
        future_features = all_data[4:6]
        y_true = all_data[3]
        tsidx = all_data[-1]
        all_preds = model((past_data, future_features, tsidx))
        y_pred = all_preds
        y_pred = y_pred[:, 0: y_true.shape[1]]
        id1 = indices[0]
        id2 = min(indices[1], y_true.shape[1])
        y_pred = y_pred[:, id1:id2]
        y_true = y_true[:, id1:id2]
        loss = train_loss(y_true, y_pred)
        all_test_loss += loss
        all_test_num += 1
        ts_count += y_true.shape[0]
        ypreds.append(y_pred)
        ytrues.append(y_true)
        if ts_count >= len(data.ts_cols):
            ts_count = 0
            ypreds = tf.concat(ypreds, axis=0)
            ytrues = tf.concat(ytrues, axis=0)
            all_y_true = set_or_concat(all_y_true, ytrues)
            all_y_pred = set_or_concat(all_y_pred, ypreds)
            ypreds = []
            ytrues = []
    return (
        all_y_true.numpy(),
        all_y_pred.numpy(),
        all_test_loss.numpy(),
        all_test_num,
    )


def training():
    """Training TS code."""
    tf.random.set_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    experiment_id = _get_random_string(8)
    logging.info('Experiment id: %s', experiment_id)

    dataset = FLAGS.dataset
    data_path = DATA_DICT[dataset]['data_path']
    freq = DATA_DICT[dataset]['freq']
    boundaries = DATA_DICT[dataset]['boundaries']

    data_df = pd.read_csv(open(data_path, 'r'))

    if FLAGS.ts_cols:
        ts_cols = DATA_DICT[dataset]['ts_cols']
        num_cov_cols = DATA_DICT[dataset]['num_cov_cols']
        cat_cov_cols = DATA_DICT[dataset]['cat_cov_cols']
    else:
        ts_cols = [col for col in data_df.columns if col != FLAGS.datetime_col]
        num_cov_cols = None
        cat_cov_cols = None
    permute = FLAGS.permute
    dtl = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col=FLAGS.datetime_col,
        num_cov_cols=num_cov_cols,
        cat_cov_cols=cat_cov_cols,
        ts_cols=np.array(ts_cols),
        train_range=[0, boundaries[0]],
        val_range=[boundaries[0], boundaries[1]],
        test_range=[boundaries[1], boundaries[2]],
        hist_len=FLAGS.hist_len,
        pred_len=FLAGS.pred_len,
        batch_size=min(FLAGS.batch_size, len(ts_cols)),
        freq=freq,
        normalize=FLAGS.normalize,
        epoch_len=FLAGS.epoch_len,
        holiday=FLAGS.holiday,
        permute=permute,
    )

    # Create model
    model_config = {
        'model_type': 'dnn',
        'hidden_dims': [FLAGS.hidden_size] * FLAGS.num_layers,
        'time_encoder_dims': [64, 4],
        'decoder_output_dim': FLAGS.decoder_output_dim,
        'final_decoder_hidden': FLAGS.final_decoder_hidden,
        'batch_size': dtl.batch_size,
    }
    model = models.TideModel(
        model_config=model_config,
        pred_len=FLAGS.pred_len,
        num_ts=len(ts_cols),
        cat_sizes=dtl.cat_sizes,
        transform=FLAGS.transform,
        layer_norm=FLAGS.layer_norm,
        dropout_rate=FLAGS.dropout_rate,
        post_data_shape=((21, 720), (8, 720), (1, 720)),
        future_features_shape=((8, 96), (1, 96)),
        tsidx_shape=(21,),
    )
    model.build()
    model.summary()
    # a = tf.TensorShape((((21, 720), (8, 720), (1, 720)), ((8, 96), (1, 96)), (21, )))
    # model.build(input_shape=[[[21, 720], [8, 720], [1, 720]], [[8, 96], [1, 96]], [21, ]])
    # model.build(input_shape=[(3, 21, 720), (2, 8, 96), (21,)])
    model.load_weights('/home/xxm/PycharmProjects/tide_demo/weights.h5')
    print('-------------')


def main(_):
    training()


if __name__ == '__main__':
    app.run(main)
