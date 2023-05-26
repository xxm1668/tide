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
from keras.callbacks import ModelCheckpoint

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
        all_preds = model((past_data, future_features, tsidx), training=True)
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
    )

    model.build()
    model.summary()

    # Compute path to experiment directory
    expt_dir = os.path.join(
        FLAGS.expt_dir,
        FLAGS.dataset + '_' + str(experiment_id) + '_' + str(FLAGS.pred_len),
    )
    os.makedirs(expt_dir, exist_ok=True)

    step = tf.Variable(0)
    # LR scheduling
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=FLAGS.learning_rate,
        decay_steps=30 * dtl.train_range[1],
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)
    summary = Summary(expt_dir)

    best_loss = np.inf
    pat = 0
    mean_loss_array = []
    iter_array = []
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # best_check_path = None
    while step.numpy() < FLAGS.train_epochs + 1:
        ep = step.numpy()
        logging.info('Epoch %s', ep)
        sys.stdout.flush()

        iterator = tqdm(dtl.tf_dataset(mode='train'), mininterval=2)
        for i, batch in enumerate(iterator):
            print(i)
            past_data = batch[:3]
            future_features = batch[4:6]
            tsidx = batch[-1]

            loss = train_step(model, past_data, future_features, batch[3], tsidx, optimizer)
            # Train metrics
            summary.update({'train/reg_loss': loss, 'train/loss': loss})
            if i % 100 == 0:
                mean_loss = summary.metric_dict['train/reg_loss'].result().numpy()
                mean_loss_array.append(mean_loss)
                iter_array.append(i)
                iterator.set_description(f'Loss {mean_loss:.4f}')
            if i == 500:
                break
        step.assign_add(1)
        for weight in model.weights:
            print(weight.name)

        # val_metrics, val_res, val_loss = eval(model, dtl, 'val', num_split=FLAGS.num_split)
        # test_metrics, test_res, test_loss = eval(model, dtl, 'test', num_split=FLAGS.num_split)
        # logging.info('Val Loss: %s', val_loss)
        # logging.info('Test Loss: %s', test_loss)
        # tracked_loss = val_metrics['rmse']
        #
        # if tracked_loss < best_loss and ep > FLAGS.min_num_epochs:
        #     print('----------保存模型----------')
        #     print(model.get_weights())
        #     model.save(os.path.join(expt_dir, 'best_model.h5'))
        #     print('----------保存模型----------')
        #     best_loss = tracked_loss
        #     pat = 0
        #
        #     with open(os.path.join(expt_dir, 'val_pred.npy'), 'wb') as fp:
        #         np.save(fp, val_res[0][:, 0: -1: FLAGS.pred_len])
        #     with open(os.path.join(expt_dir, 'val_true.npy'), 'wb') as fp:
        #         np.save(fp, val_res[1][:, 0: -1: FLAGS.pred_len])
        #
        #     with open(os.path.join(expt_dir, 'test_pred.npy'), 'wb') as fp:
        #         np.save(fp, test_res[0][:, 0: -1: FLAGS.pred_len])
        #     with open(os.path.join(expt_dir, 'test_true.npy'), 'wb') as fp:
        #         np.save(fp, test_res[1][:, 0: -1: FLAGS.pred_len])
        #     with open(os.path.join(expt_dir, 'test_metrics.json'), 'w') as fp:
        #         json.dump(test_metrics, fp)
        #     logging.info('saved best result so far at %s', expt_dir)
        #     logging.info('Test metrics: %s', test_metrics)
        # else:
        #     pat += 1
        #     if pat > FLAGS.patience:
        #         logging.info('Early stopping')
        #         break

        summary.write(step=step.numpy())

    model.save_weights('/home/xxm/PycharmProjects/tide_demo/OUTPUT/best_model.h5')


def main(_):
    training()


if __name__ == '__main__':
    app.run(main)
