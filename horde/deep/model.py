from collections import defaultdict

import tensorflow as tf

from .layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, \
    MultiHotEncoding, GraphConvolutionRecurrent, GraphNeighborsDecoding
from . import inits

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

'''
Input : Batch of context node-pairs
'''
class TimeInvariantNode(Model):
    def __init__(self, placeholders, adj_mat, feature, num_feats, num_nodes, **kwargs):
        super(TimeInvariantNode, self).__init__(**kwargs)
        self.adj_mat = tf.cast(tf.SparseTensor(adj_mat[0], adj_mat[1], adj_mat[2]), dtype=tf.float32)
        self.inputs = tf.cast(tf.SparseTensor(feature[0], feature[1], feature[2]), dtype=tf.float32)
        self.input_dim = num_feats
        self.num_nodes = num_nodes
        self.dropout = placeholders['ti_dropout']

        self.build()

    def _build(self):
        self.first_embeds = GraphConvolutionSparseMulti(
            input_dim = self.input_dim, output_dim = FLAGS.hidden1,
            adj_mat = self.adj_mat, num_nodes = self.num_nodes, 
            act = tf.nn.relu, dropout = self.dropout,
            logging = self.logging)(self.inputs)

        self.second_embeds = GraphConvolutionMulti(
            input_dim = FLAGS.hidden1, output_dim = FLAGS.hidden2,
            adj_mat = self.adj_mat,
            act = tf.nn.tanh, dropout = self.dropout,
            logging = self.logging)(self.first_embeds)

        self.ti_embeds = self.second_embeds

        with tf.variable_scope('%s_vars' % self.name):
            self.vars['ctx_embeds'] = inits.weight_variable_glorot(self.input_dim, FLAGS.hidden2, name='ctx_embeds')
        self.ctx_embeds = self.vars['ctx_embeds']

'''
Input : Batch of visit sequences
'''
class TimeVariantNode(Model):
    def __init__(self, placeholders, adj_mat, feature, degrees, num_steps, num_nodes, **kwargs):
        super(TimeVariantNode, self).__init__(**kwargs)
        self.adj_mat = tf.cast(tf.SparseTensor(adj_mat[0], adj_mat[1], adj_mat[2]), dtype=tf.float32)
        self.feature = feature
        self.degrees = degrees
        self.num_steps = num_steps
        self.num_nodes = num_nodes

        self.inputs = placeholders['batch_tv_seq']
        self.dropout = placeholders['tv_dropout']

        self.build()

    def _build(self):
        #self.multihots = MultiHotEncoding(
        #    num_nodes = self.num_nodes, logging = self.logging)(self.inputs)

        self.tv_embeds = GraphConvolutionRecurrent(
            feature = self.feature, degrees = self.degrees,
            num_nodes=self.num_nodes,
            input_dim = FLAGS.hidden1, output_dim = FLAGS.hidden2,
            act = lambda x: x, dropout = self.dropout, 
            logging = self.logging)(self.inputs)

        '''
        self.logits = GraphNeighborsDecoding(
            softmax_weight = self.softmax_weight, 
            input_dim = FLAGS.hidden2, output_dim = self.num_nodes,
            num_steps = self.num_steps, logging = self.logging)(self.tv_embeds)

        self.softmax = tf.nn.softmax(self.logits, axis=2)
        '''
