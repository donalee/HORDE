import tensorflow as tf

from . import inits

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties    
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.is_sparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mat,
                 num_nodes, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mat = adj_mat
        self.act = act
        self.is_sparse = True
        self.num_nodes = num_nodes
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = inits.weight_variable_glorot(input_dim, output_dim, name='weights')

    def _call(self, inputs):
        x = dropout_sparse(inputs, 1-self.dropout, self.num_nodes)
        xw = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        axw = tf.sparse_tensor_dense_matmul(self.adj_mat, xw)
        outputs = self.act(axw)
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        return outputs


class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mat = adj_mat
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = inits.weight_variable_glorot(input_dim, output_dim, name='weights')

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        xw = tf.matmul(x, self.vars['weights'])
        axw = tf.sparse_tensor_dense_matmul(self.adj_mat, xw)
        outputs = self.act(axw)
        outputs = tf.nn.l2_normalize(outputs, axis=1)
        return outputs

# This layer is deprecated due to the memory-inefficiency reason
class MultiHotEncoding(MultiLayer):
    def __init__(self, num_nodes, **kwargs):
        super(MultiHotEncoding, self).__init__(**kwargs)
        self.num_nodes = num_nodes

    def _call(self, inputs):
        onehots = tf.one_hot(inputs, depth=self.num_nodes, on_value=1.0, off_value=0.0)
        outputs = tf.reduce_sum(onehots, axis=2)
        return outputs

class GraphConvolutionRecurrent(MultiLayer):
    """Basic LSTM layer for time-vairant nodes in time-evolving graph"""
    def __init__(self, feature, degrees, num_nodes, input_dim, output_dim, forget_bias=1., dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionRecurrent, self).__init__(**kwargs)
        self.feature = feature
        self.degrees = degrees
        self.num_nodes = num_nodes 
        self.hidden_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = inits.weight_variable_glorot(input_dim, output_dim, name='weights')
            self.vars['cell'] = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias = forget_bias)
            self.vars['cell'] = tf.contrib.rnn.DropoutWrapper(self.vars['cell'], output_keep_prob = 1-dropout)

    def _length(self, inputs):
        used = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2))
        length = tf.reduce_sum(used, 1)
        legnth = tf.cast(length, tf.int32)
        return length

    def _call(self, inputs):
        tv_degrees = tf.clip_by_value(tf.reduce_sum(inputs, axis=2), 1, self.num_nodes)
        tv_degrees = tf.tile(tf.expand_dims(tv_degrees, axis=2), [1, 1, self.num_nodes])
        tv_degrees = tf.sqrt(tf.to_float(tv_degrees))
        ti_degrees = tf.sqrt(tf.to_float(tf.constant(self.degrees)))

        norm_inputs = tf.to_float(inputs)/(tv_degrees*ti_degrees)

        x = tf.nn.dropout(self.feature, 1-self.dropout)
        xw = tf.matmul(x, self.vars['weights'])
        axw = tf.tensordot(norm_inputs, xw, axes=1)
        outputs, self.state = tf.nn.dynamic_rnn(
            self.vars['cell'], axw, dtype=tf.float32, 
            sequence_length = self._length(axw))
        outputs = self.act(outputs)
        #outputs = tf.nn.l2_normalize(outputs, axis=1)
        # outputs.shape => (batch_size * num_steps, hidden_dim)
        return outputs

class GraphNeighborsDecoding(MultiLayer):
    """Decoding layer for predicting its neighbor nodes in the next timestep graph"""
    def __init__(self, softmax_weight, input_dim, output_dim, num_steps, **kwargs):
        super(GraphNeighborsDecoding, self).__init__(**kwargs)
        self.softmax_weight = softmax_weight
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = inits.weight_variable_glorot(input_dim, output_dim, name='weight')
            self.vars['bias'] = inits.bias_variable_glorot(output_dim, name='bias')

    def _call(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.input_dim])
        if self.softmax_weight is None:
            outputs = tf.nn.xw_plus_b(inputs, self.vars['weights'], self.vars['bias'])
        else:
            outputs = tf.nn.xw_plus_b(inputs, tf.transpose(self.softmax_weight), self.vars['bias'])
        outputs = tf.reshape(outputs, [-1, self.num_steps, self.output_dim])
        return outputs





