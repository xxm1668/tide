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

"""This file contains the TiDE model  code."""

from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

EPS = 1e-7

train_loss = keras.losses.MeanSquaredError()


class MLPResidual(keras.layers.Layer):
    """Simple one hidden state residual network."""

    def __init__(
            self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0, name=None
    ):
        super(MLPResidual, self).__init__()
        self.lin_a = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            name=name + '_lin_a',
            weights=name + '_lin_a_weights',
        )
        self.lin_b = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            name=name + '_lin_b',
            weights=name + '_lin_a_weights',
        )
        self.lin_res = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            name=name + '_lin_res',
        )
        if layer_norm:
            self.lnorm = tf.keras.layers.LayerNormalization()
        self.layer_norm = layer_norm
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape=None):
        """Build method."""
        self.lin_a.build((720, 12))
        self.lin_b.build((720, 64))
        self.lin_res.build((720, 12))
        self.built = True

    def call(self, inputs):
        """Call method."""
        h_state = self.lin_a(inputs)
        out = self.lin_b(h_state)
        out = self.dropout(out)
        res = self.lin_res(inputs)
        if self.layer_norm:
            return self.lnorm(out + res)
        return out + res


def _make_dnn_residual(hidden_dims, layer_norm=False, dropout_rate=0.0, name=None):
    """Multi-layer DNN residual model."""
    if len(hidden_dims) < 2:
        return keras.layers.Dense(
            hidden_dims[-1],
            activation=None,
        )
    layers = []
    for i, hdim in enumerate(hidden_dims[:-1]):
        layers.append(
            MLPResidual(
                hdim,
                hidden_dims[i + 1],
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
                name=name + '_layer_{}'.format(i),
            )
        )
    return keras.Sequential(layers, name=name)


class TideModel(keras.Model):
    """Main class for multi-scale DNN model."""

    def __init__(
            self,
            model_config,
            pred_len,
            cat_sizes,
            num_ts,
            transform=False,
            cat_emb_size=4,
            layer_norm=False,
            dropout_rate=0.0,
            post_data_shape=None,
            future_features_shape=None,
            tsidx_shape=None,
    ):
        """Tide model.

        Args:
          model_config: configurations specific to the model.
          pred_len: prediction horizon length.
          cat_sizes: number of categories in each categorical covariate.
          num_ts: number of time-series in the dataset
          transform: apply reversible transform or not.
          cat_emb_size: embedding size of categorical variables.
          layer_norm: use layer norm or not.
          dropout_rate: level of dropout.
        """
        super().__init__()
        self.model_config = model_config
        self.transform = transform
        if self.transform:
            self.affine_weight = self.add_weight(
                name='affine_weight',
                shape=(num_ts,),
                initializer='ones',
                trainable=True,
            )

            self.affine_bias = self.add_weight(
                name='affine_bias',
                shape=(num_ts,),
                initializer='zeros',
                trainable=True,
            )
        self.pred_len = pred_len
        self.encoder = _make_dnn_residual(
            model_config.get('hidden_dims'),
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            name='encoder',
        )
        self.decoder = _make_dnn_residual(
            model_config.get('hidden_dims')[:-1]
            + [
                model_config.get('decoder_output_dim') * self.pred_len,
            ],
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            name='decoder',
        )
        self.linear = tf.keras.layers.Dense(
            self.pred_len,
            activation=None,
        )
        self.time_encoder = _make_dnn_residual(
            model_config.get('time_encoder_dims'),
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            name='time_encoder',
        )
        self.final_decoder = MLPResidual(
            hidden_dim=model_config.get('final_decoder_hidden'),
            output_dim=1,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            name='final_decoder',
        )
        self.cat_embs = []
        for cat_size in cat_sizes:
            self.cat_embs.append(
                tf.keras.layers.Embedding(input_dim=cat_size, output_dim=cat_emb_size)
            )
        self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)
        self.post_data_shape = post_data_shape
        self.future_features_shape = future_features_shape
        self.tsidx_shape = tsidx_shape

    def _assemble_feats(self, feats, cfeats):
        """assemble all features."""
        all_feats = [feats]
        for i, emb in enumerate(self.cat_embs):
            all_feats.append(tf.transpose(emb(cfeats[i, :])))
        return tf.concat(all_feats, axis=0)

    def build(self, input_shape=None):
        """Build the model."""
        self.encoder.build((21, 4000))
        self.decoder.build((21, 256))
        self.linear.build((21, 720))
        self.time_encoder.build((720, 12))
        self.final_decoder.build((21, 96, 12))
        self.ts_embs.build((21,))
        self.cat_embs[0].build((720, 4))
        self.built = True

    def call(self, inputs):
        """Call function that takes in a batch of training data and features."""
        past_data = inputs[0]
        future_features = inputs[1]
        bsize = past_data[0].shape[0]
        tsidx = inputs[2]
        past_feats = self._assemble_feats(past_data[1], past_data[2])
        future_feats = self._assemble_feats(future_features[0], future_features[1])
        past_ts = past_data[0]
        if self.transform:
            affine_weight = tf.gather(self.affine_weight, tsidx)
            affine_bias = tf.gather(self.affine_bias, tsidx)
            batch_mean = tf.math.reduce_mean(past_ts, axis=1)
            batch_std = tf.math.reduce_std(past_ts, axis=1)
            batch_std = tf.where(
                tf.math.equal(batch_std, 0.0), tf.ones_like(batch_std), batch_std
            )
            past_ts = (past_ts - batch_mean[:, None]) / batch_std[:, None]
            past_ts = affine_weight[:, None] * past_ts + affine_bias[:, None]
        encoded_past_feats = tf.transpose(
            self.time_encoder(tf.transpose(past_feats))
        )
        encoded_future_feats = tf.transpose(
            self.time_encoder(tf.transpose(future_feats))
        )
        enc_past = tf.repeat(tf.expand_dims(encoded_past_feats, axis=0), bsize, 0)
        enc_past = tf.reshape(enc_past, [bsize, -1])
        enc_fut = tf.repeat(
            tf.expand_dims(encoded_future_feats, axis=0), bsize, 0
        )  # batch x fdim x H
        enc_future = tf.reshape(enc_fut, [bsize, -1])
        residual_out = self.linear(past_ts)
        ts_embs = self.ts_embs(tsidx)
        encoder_input = tf.concat([past_ts, enc_past, enc_future, ts_embs], axis=1)
        encoding = self.encoder(encoder_input)
        decoder_out = self.decoder(encoding)
        decoder_out = tf.reshape(
            decoder_out, [bsize, -1, self.pred_len]
        )  # batch x d x H
        final_in = tf.concat([decoder_out, enc_fut], axis=1)
        out = self.final_decoder(tf.transpose(final_in, (0, 2, 1)))  # B x H x 1
        out = tf.squeeze(out, axis=-1)
        out += residual_out
        if self.transform:
            out = (out - affine_bias[:, None]) / (affine_weight[:, None] + EPS)
            out = out * batch_std[:, None] + batch_mean[:, None]
        return out
