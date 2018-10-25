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

    def process_weights_and_bias(self, rnn_weights):
        raise ValueError("not implemented")

    def process_input_x(self, rnn_props, rnn_scope_name):
        raise ValueError("not implemented")

    def process_var_init_nodes(self, rnn_props):
        raise ValueError("not implemented")

    def process_seq_length(self, rnn_props, seq_len_input_node):
        raise ValueError("not implemented")

    def create_rnn_node(self, rnn_props):
        raise ValueError("not implemented")
