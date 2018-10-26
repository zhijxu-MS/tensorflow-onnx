# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for gru."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from termcolor import cprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from backend_test_base import Tf2OnnxBackendTestBase

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test


def save_graph(output_names_with_port):
    from tensorflow.python.ops import variables as variables_lib

    with tf.Session() as sess:
        variables_lib.global_variables_initializer().run()
        # expected = sess.run(output_dict, feed_dict=feed_dict)
        output_name_without_port = [n.split(':')[0] for n in output_names_with_port]
        #graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
        #                                                         output_name_without_port)

        tf.train.write_graph(sess.graph_def, logdir=r'C:\Users\zhijxu\Desktop\GRU', name="ori.pb", as_text=True)
        saver = tf.train.Saver()
        saver.save(sess, r'C:\Users\zhijxu\Desktop\GRU\ckpt')


def save_tensorboard_with_graph():
    from tensorflow.python.platform import gfile

    tf.reset_default_graph()
    model_filename = r'C:\Users\zhijxu\Desktop\GRU\frozen.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    sess = tf.Session()
    logdir = r'C:\Users\zhijxu\Desktop\GRU'
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(sess.graph)
    train_writer.close()
    cprint("save graph in tensorboard done", "green")


class GRUTests(Tf2OnnxBackendTestBase):
    def test_single_dynamic_gru(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")

        # no scope
        cell = rnn.GRUCell(
            units,
            activation=None)
        outputs, cell_state = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32)

        _ = tf.identity(outputs, name="output")
        _ = tf.identity(cell_state, name="cell_state")

        input_names_with_port = ["input_1:0"]
        feed_dict = {"input_1:0": x_val}
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-05)

    def test_multiple_dynamic_gru_with_parameters(self):
        units = 5
        batch_size = 6
        x_val = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
        x_val = np.stack([x_val] * batch_size)

        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        _ = tf.placeholder(tf.float32, x_val.shape, name="input_2")
        initializer = init_ops.constant_initializer(0.5)

        lstm_output_list = []
        lstm_cell_state_list = []
        if True:
            # no scope
            cell = rnn.GRUCell(
                units,
                activation=None)
            outputs, cell_state = tf.nn.dynamic_rnn(
                cell,
                x,
                dtype=tf.float32)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)

        if True:
            # given scope
            cell = rnn.GRUCell(
                units,
                activation=None)
            with variable_scope.variable_scope("root1") as scope:
                outputs, cell_state = tf.nn.dynamic_rnn(
                    cell,
                    x,
                    dtype=tf.float32,
                    sequence_length=[4, 4, 4, 4, 4, 4],
                    scope=scope)
            lstm_output_list.append(outputs)
            lstm_cell_state_list.append(cell_state)

        _ = tf.identity(lstm_output_list, name="output")
        _ = tf.identity(lstm_cell_state_list, name="cell_state")

        feed_dict = {"input_1:0": x_val}
        input_names_with_port = ["input_1:0"]
        output_names_with_port = ["output:0", "cell_state:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port, rtol=1e-5)


if __name__ == '__main__':
    Tf2OnnxBackendTestBase.trigger(GRUTests)
