# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for fold_tf_const."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib import rnn

from backend_test_base import Tf2OnnxBackendTestBase
from common import unittest_main


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


class FoldTfConstTests(Tf2OnnxBackendTestBase):

    def test_single_dynamic_lstm(self):
        units = 5
        seq_len = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * seq_len)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # no scope
        cell = rnn.LSTMCell(
            units,
            initializer=initializer)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            time_major=True,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}

        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port,
                           rtol=1e-06, constant_fold=False)

    def test_dynamic_bilstm(self, state_is_tuple=True):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        initializer = init_ops.constant_initializer(0.5)

        # bilstm, no scope
        cell1 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)  # state_is_tuple will impact Pack node (for cell_state)'s usage pattern
        cell2 = rnn.LSTMCell(
            units,
            initializer=initializer,
            state_is_tuple=state_is_tuple)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell1,
            cell2,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port,
                           rtol=1e-06, constant_fold=False)


if __name__ == '__main__':
    unittest_main()
