# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.fold_const_using_tf - find nodes in tensorflow graph that can be fold, namely treated as a const;
and run tensorflow to get their values, and replace them in tf2onnx graph
"""
import copy

import tensorflow as tf
from tensorflow.python.ops import gen_control_flow_ops
from tf2onnx import logging


logger = logging.getLogger(__name__)


# pylint: disable=logging-not-lazy,missing-docstring


def const_nodes_and_values_in_tf(tf_graph):
    """
    :return: a dict, whose key:value is tensor_name : tensor_value;
    and dtype of tenor value is defined in numpy, such as numpy.int32, numpy.ndarray.
    result will only contain the tensors that aren't const but can be treat as a const, and are consumed by non-const.
    algorithm:
    1. find nodes can be const: if node's inputs are all can be const, then it can be a const also
       and some special ops are ignored such as control flow ops.
    2. find nodes whose consumer is not const.
    3. call tf_session.run to get their values.
    """
    nodes_can_be_const = find_nodes_can_be_const(tf_graph)
    nodes_can_be_const = find_nodes_consumed_by_non_const(nodes_can_be_const)
    res = get_node_vals_used_tf(tf_graph, nodes_can_be_const)
    return res


def repalce_node_with_const(const_nodes_value_dict, onnx_graph):
    for tensor_name, value in const_nodes_value_dict.items():
        node_to_remove = onnx_graph.get_node_by_output(tensor_name)
        if node_to_remove:  # if node has multiple outputs, then make sure it's deleted only once
            onnx_graph.remove_node(node_to_remove.name)
        onnx_graph.make_const(tensor_name, value)


# helper functions
_control_flow_ops = list(gen_control_flow_ops._op_def_lib._ops.keys())  # pylint: disable=protected-access


def is_special_ops(tf_op):
    if tf_op.type in _control_flow_ops:
        return True
    if tf_op.type == "PlaceholderWithDefault":
        return True
    if tf_op.type.startswith("TensorArray"):
        return True
    random_op_prefix = ["Random", "Dropout"]
    for prefix in random_op_prefix:
        if tf_op.type.startswith(prefix):
            return True

    return False


def find_node_consumers(tf_node):
    res = []
    for out in tf_node.outputs:
        out_consumers = out.consumers()
        res.extend(out_consumers)  # it's ok to contain duplicated nodes
    return res


def find_nodes_can_be_const(tf_graph):
    """
    how to determine a node can be a const:
    1 put all nodes that are const into set x
    2 iterate node in set x, to see if node's consumers can be const: if consumer's inputs
    are all can be const then the consumer can be const so put the consumer into set y
    3 set x = y, loop to step 2 until y is empty
    :return: set of nodes that can be const, please be note that const node is not included.
    """
    def node_can_be_const(tf_node, nodes_can_be_const):
        if is_special_ops(tf_node):
            return False
        input_nodes = set(input_node._op for input_node in tf_node.inputs)  # pylint: disable=protected-access

        return input_nodes.issubset(nodes_can_be_const)

    const_nodes = []
    for node in tf_graph.get_operations():
        if node.type == "Const":
            const_nodes.append(node)

    nodes_can_be_const = set(copy.copy(const_nodes))
    while True:
        new_found = set()
        for node in nodes_can_be_const:
            consumers = find_node_consumers(node)
            for candidate in consumers:
                if candidate in nodes_can_be_const:
                    continue  # avoid graph contains loop structure
                if node_can_be_const(candidate, nodes_can_be_const):
                    new_found.add(candidate)
        if not new_found:
            break
        else:
            nodes_can_be_const.update(new_found)

    nodes_can_be_const.difference_update(const_nodes)
    return nodes_can_be_const


def find_nodes_consumed_by_non_const(nodes_can_be_const):
    def is_consumed_by_non_const(node):
        consumers = set(find_node_consumers(node))
        return not consumers.issubset(nodes_can_be_const)

    res = list(filter(is_consumed_by_non_const, nodes_can_be_const))
    return res


def get_node_vals_used_tf(tf_graph, nodes_can_be_const):
    tensor_names = []
    for node in nodes_can_be_const:
        for out in node.outputs:
            tensor_names.append(out.name)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf_graph) as sess:
        res = {}
        for tensor_name in tensor_names:
            try:
                val = sess.run(tensor_name)
            except tf.errors.InvalidArgumentError as ex:  # pylint: disable=unused-variable
                continue
            res[tensor_name] = val
    return res


# debug function
def get_names_of_nodes(nodes):
    res = sorted(node.name for node in nodes)
    return res
