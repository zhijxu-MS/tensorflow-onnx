# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.gru_rewriter - gru support
"""

from __future__ import division
from __future__ import print_function

import sys

from tf2onnx.rewriter.unit_rewriter_base import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.gru_rewriter")


class GRUUnitRewriter(UnitRewriterBase):
    def __init__(self, g):
        super(GRUUnitRewriter, self).__init__(g)
        self.switch_checkers = {
            # True means we need parse its initial value in later logic.
            # in tensorflow, switch is a good op that we can use to trace other ops that needed
            "state": (self._state_switch_check, self._connect_gru_state_to_graph, True),
            "output": (self._output_switch_check, self._connect_gru_output_to_graph, False),
        }

    def run(self):
        return super(GRUUnitRewriter, self).run(RNNUnitType.GRUCell)

    def get_rnn_scope_name(self, match):
        # take the cell output and go up 3 levels to find the scope:
        # name of h is like root/while/gru_cell/mul_2
        # root is the dynamic rnn's scope name.
        # root/while/gru_cell is cell's scope name
        h_node = match.get_op("cell_output")
        parts = h_node.name.split('/')
        rnn_scope_name = '/'.join(parts[0:-3])
        return rnn_scope_name

    def get_weight_and_bias(self, match):
        gate_kernel = get_weights_from_const_node(match.get_op("gate_kernel"))
        gate_bias = get_weights_from_const_node(match.get_op("gate_bias"))
        hidden_kernel = get_weights_from_const_node(match.get_op("hidden_kernel"))
        hidden_bias = get_weights_from_const_node(match.get_op("hidden_bias"))
        if not all([gate_kernel, gate_bias, hidden_kernel, hidden_bias]):
            log.error("rnn weights check failed, skip")
            sys.exit(-1)
        else:
            log.debug("find needed weights")
            res = {'gate_kernel': gate_kernel,
                   "gate_bias": gate_bias,
                   "hidden_kernel": hidden_kernel,
                   "hidden_bias": hidden_bias}
            return res

    @staticmethod
    def _state_switch_check(enter_target_node_input_id, identity_consumers, match):
        concat_nodes = [c for c in identity_consumers if c == match.get_op("cell_inputs")]
        if len(concat_nodes) == 1:
            log.debug("find state initializer value at " + enter_target_node_input_id)
            return enter_target_node_input_id
        else:
            log.debug(str(len(concat_nodes)) + "Concat matching found, cannot identify state initializer")
            return None

    def _connect_gru_state_to_graph(self):
        pass

    def _output_switch_check(self, enter_target_node_input_id, identity_consumers, match):
        raise ValueError("not implemented")

    def _connect_gru_output_to_graph(self):
        pass

    def get_rnn_input_blacklist(self, rnn_weights, rnn_props):
        var_init_nodes = []
        for _, init_input_id in rnn_props.var_initializers.items():
            init_node = self.g.get_node_by_name(init_input_id)
            var_init_nodes.append(init_node)
            self.must_keep_nodes.append(init_node)
        blacklist_inputs = []
        blacklist_inputs.extend(var_init_nodes)
        # weight/bias inputs, and c/h initializer are dynamic_rnn/LSTMCell's parameters.
        # we will use them to filter out the dynamic_rnn's input tensor.
        for _, value in rnn_weights.items():
            blacklist_inputs.append(value.node)

        return blacklist_inputs

    def process_weights_and_bias(self, rnn_weights, rnn_props):
        """
        why split the data in this way should refer to code of tensorflow GRU cell and official document of ONNX GRU
        """
        # from code of tensorflow GRU cell, it can be known that shape of hidden_kernel(or candidate_kernel)
        # is (input_size+hidden_unit, hidden_unit)
        hidden_size = rnn_weights["hidden_kernel"].value.shape[1]
        input_size = rnn_weights["hidden_kernel"].value.shape[0] - hidden_size
        weight_dtype = rnn_weights["hidden_kernel"].dtype
        bias_dtype = rnn_weights["hidden_bias"].dtype
        # below code will use same notation as ONNX document
        # z means update gate, r means reset gate, h means hidden gate;
        # at this time weights of gate include input and state, will split it next
        r_kernel, z_kernel = np.split(rnn_weights["gate_kernel"].value, [hidden_size], axis=1)
        h_kernel = rnn_weights["hidden_kernel"].value
        r_bias, z_bias = np.split(rnn_weights["gate_bias"].value, [hidden_size], axis=0)
        h_bias = rnn_weights["hidden_bias"].value
        # ONNX GRU split weights of input and state, so have to split *_kernel
        input_r_kernel, state_r_kernel = np.split(r_kernel, [input_size], axis=0)
        input_z_kernel, state_z_kernel = np.split(z_kernel, [input_size], axis=0)
        input_h_kernel, state_h_kernel = np.split(h_kernel, [input_size], axis=0)
        W_zrh = np.concatenate((input_z_kernel, input_r_kernel, input_h_kernel), axis=1)
        R_zrh = np.concatenate((state_z_kernel, state_r_kernel, state_h_kernel), axis=1)
        # transpose weight matrix
        W_zrh = np.transpose(np.expand_dims(W_zrh, axis=0), axes=(0, 2, 1))
        R_zrh = np.transpose(np.expand_dims(R_zrh, axis=0), axes=(0, 2, 1))
        W_zrh = W_zrh.astype(weight_dtype)
        R_zrh = R_zrh.astype(weight_dtype)
        assert W_zrh.shape == (1, 3*hidden_size, input_size)
        assert R_zrh.shape == (1, 3*hidden_size, hidden_size)
        Wb_zrh = np.concatenate((z_bias, r_bias, h_bias), axis=0)
        # tf don't have bias for state, so use 0 instead
        zero = np.zeros_like(z_bias)
        Rb_zrh = np.concatenate((zero, zero, zero), axis=0)
        B_zrh = np.concatenate((Wb_zrh, Rb_zrh), axis=0)
        B_zrh = np.expand_dims(B_zrh, axis=0)
        B_zrh = B_zrh.astype(bias_dtype)
        assert B_zrh.shape == (1, 6*hidden_size)
        # create const ONNX node
        w_name = utils.make_name("W")
        w_node = self.g.make_const(w_name, W_zrh, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = self.g.make_const(r_name, R_zrh, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = self.g.make_const(b_name, B_zrh, skip_conversion=True)

        rnn_props.input_size = input_size
        rnn_props.hidden_size = hidden_size
        rnn_props.onnx_input_ids["W"] = w_node.output[0]
        rnn_props.onnx_input_ids["R"] = r_node.output[0]
        rnn_props.onnx_input_ids["B"] = b_node.output[0]

    def process_var_init_nodes(self, rnn_props):
        raise ValueError("not implemented")


    def create_rnn_node(self, rnn_props):
        raise ValueError("not implemented")
