import tensorflow as tf
import numpy as np

from .layers import GraphNeighborsDecoding
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS

class TINodeOptimizer(object):
    def __init__(self, ti_embeds, ctx_embeds, degrees, negsample_size, placeholders, batch_size=100):
        self.ti_embeds = ti_embeds
        self.ctx_embeds = ctx_embeds
        self.degrees = degrees
        self.negsample_size = negsample_size
        self.batch_size = batch_size

        self.inputs = placeholders['batch_ti_pair']
        self.src_inputs = tf.reshape(gather_cols(self.inputs, [0]), [1, self.batch_size])
        self.dst_inputs = tf.reshape(gather_cols(self.inputs, [1]), [1, self.batch_size])
        
        # negative node sampling
        labels = tf.reshape(tf.cast(self.dst_inputs, dtype=tf.int64), [self.batch_size, 1])
        neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes = labels,
            num_true = 1,
            num_sampled = self.batch_size * self.negsample_size,
            unique = False,
            range_max = len(self.degrees),
            distortion = 0.75,
            unigrams = self.degrees.tolist())
        self.neg_samples = tf.reshape(neg_samples, [self.negsample_size, self.batch_size])

        self.outputs = self.batch_affinity(self.src_inputs, self.dst_inputs)
        self.neg_outputs = self.batch_affinity(self.src_inputs, self.neg_samples)

        self._build()

    def batch_affinity(self, src_inputs, dst_inputs):
        src_embeds = tf.nn.embedding_lookup(self.ti_embeds, src_inputs)
        dst_embeds = tf.nn.embedding_lookup(self.ctx_embeds, dst_inputs)

        aff = tf.multiply(src_embeds, dst_embeds)
        aff = tf.reduce_sum(aff, axis=2)
        return aff

    def _build(self):
        #self.cost = self._softmax_loss(self.src_inputs, self.dst_inputs)
        self.cost = self._logsigmoid_loss_(self.outputs, self.neg_outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)

    def _logsigmoid_loss_(self, pos_aff, neg_aff):
        """Log-sigmoid optimization."""
        logsigmoid_pos_aff = tf.math.log(tf.sigmoid(pos_aff) + tf.constant(value=1e-8))
        logsigmoid_neg_aff = tf.math.log(tf.sigmoid(-1.0*neg_aff) + tf.constant(value=1e-8))
        loss = -1.0 * (tf.reduce_sum(logsigmoid_pos_aff) + tf.reduce_sum(logsigmoid_neg_aff))
        return loss

    def _softmax_loss(self, src_inputs, dst_inputs):
        src_embeds = tf.nn.embedding_lookup(self.ti_embeds, src_inputs)
        dst_embeds = tf.nn.embedding_lookup(self.ctx_embeds, dst_inputs)

        norm = tf.reduce_sum(tf.math.exp(tf.matmul(tf.squeeze(src_embeds), tf.transpose(self.ctx_embeds))), axis=1)
        norm = tf.expand_dims(norm, 0)

        aff = tf.math.exp(tf.reduce_sum(tf.multiply(src_embeds, dst_embeds), axis=2))
        aff = aff / (norm + tf.constant(value=1e-8))

        loss = -1.0 * tf.reduce_sum(tf.math.log(aff + tf.constant(value=1e-8)))
        return loss

class TVNodeOptimizer(object):
    def __init__(self, tv_embeds, ctx_embeds, labels, num_nodes, num_steps):

        self.tv_embeds = tv_embeds
        self.labels = labels

        self.logits = GraphNeighborsDecoding(
            softmax_weight = ctx_embeds,
            input_dim = FLAGS.hidden2, output_dim = num_nodes,
            num_steps = num_steps)(tv_embeds)

        self.softmax = tf.nn.softmax(self.logits, axis=2)

        self._build()

    def _build(self):
        self.cost = self._softmax_xent_loss(self.labels[:,1:,:], self.softmax[:,:-1,:])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)

    def _softmax_xent_loss(self, labels, softmax):
        """Cross-entropy optimization."""
        mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))
        softmax = softmax + tf.constant(value=1e-8)

        labels = tf.sign(labels)
        cross_entropy = tf.to_float(labels) * tf.log(softmax)
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        cross_entropy *= tf.to_float(mask)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        #cross_entropy /= tf.reduce_sum(mask, axis=1)
        loss = tf.reduce_sum(cross_entropy)
        return loss

class UnifiedOptimizer(object):
    def __init__(self, ti_model, tv_model, ti_degrees, negsample_size, placeholders, ti_batch_size, tv_batch_size):
        self.ti_model, self.tv_model = ti_model, tv_model

        self.ti_opt = TINodeOptimizer(
            ti_embeds = ti_model.ti_embeds,
            ctx_embeds = ti_model.ti_embeds,
            degrees = ti_degrees,
            negsample_size = negsample_size,
            placeholders = placeholders,
            batch_size = ti_batch_size)

        self.tv_opt = TVNodeOptimizer(
            tv_embeds = tv_model.tv_embeds,
            ctx_embeds = ti_model.ti_embeds,
            labels = tv_model.inputs,
            num_nodes = tv_model.num_nodes,
            num_steps = tv_model.num_steps)

        self._build()

    def _build(self):
        self.cost = self.ti_opt.cost + self.tv_opt.cost
        self.cost += FLAGS.weight_decay * sum([tf.nn.l2_loss(var) for var in self.ti_model.vars.values()])
        self.cost += FLAGS.weight_decay * sum([tf.nn.l2_loss(var) for var in self.tv_model.vars.values()])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)


def gather_cols(params, indices, name=None):
    with tf.name_scope(name, "gather_cols", [params, indices]) as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(
            tf.gather(p_flat, i_flat), [p_shape[0], -1])
