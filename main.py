from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os, sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import argparse

from horde.deep.optimizer import TINodeOptimizer, TVNodeOptimizer, UnifiedOptimizer
from horde.deep.model import TimeInvariantNode, TimeVariantNode
from horde.deep.minibatch import CtxPairMinibatchIterator, SeqMinibatchIterator
from horde.utility import metrics, preprocessing

###########################################################
#
# Placeholders and settings
#
###########################################################

np.random.seed(0)

def construct_placeholders(max_length, num_nodes):
    placeholders = {
        'batch_ti_pair': tf.placeholder(tf.int32, [None, 2], name='batch_ti_pair'),
        'batch_tv_seq': tf.placeholder(tf.int32, [None, max_length, num_nodes], name='batch_tv_seq'),
        'ti_dropout': tf.placeholder_with_default(0., shape=(), name='ti_dropout'),
        'tv_dropout': tf.placeholder_with_default(0., shape=(), name='tv_dropout')
    }
    return placeholders

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', './', 'The path to the input numpy files')
flags.DEFINE_string('output_dir', './', 'The path to the output embedding numpy files')
flags.DEFINE_string('label_file', '', 'The path to the numpy file containing category labels of event nodes (optional)')
flags.DEFINE_integer('vector_size', 256, 'The  size of the final representation vectors')
flags.DEFINE_integer('negsample_size', 1, 'The number of negative context pairs per positive context pair')
flags.DEFINE_float('learning_rate', 0.001, 'The initial learning rate for the ADAM optimizer')
flags.DEFINE_float('ti_dropout', 0.3, 'The dropout rate for time-invariant node vectors')
flags.DEFINE_float('tv_dropout', 0.3, 'The dropout rate for time-variant node vectors')
flags.DEFINE_integer('ti_batch_size', 512, 'The size of a single mini-batch for time-invariant node pairs')
flags.DEFINE_integer('tv_batch_size', 32, 'The size of a single mini-batch for time-variant nod sequences')
flags.DEFINE_float('weight_decay', 0.001, 'The coefficient of L2 regularization on all the weight parameters')
flags.DEFINE_integer('n_iters', 200000, 'The total number of mini-batches for training')
flags.DEFINE_integer('n_printiters', 2000, 'The number of mini-batches for evaluating the model and printing outputs')
flags.DEFINE_integer('recall_at', 30, 'The k value in Recall@k for subsequent event prediction')
flags.DEFINE_boolean('gpu', False, 'Enable to use GPU for training, instead of CPU')
flags.DEFINE_integer('gpu_devidx', 0, 'The device index of the target GPU (in case that multiple GPUs are available)')

flags.DEFINE_integer('hidden1', FLAGS.vector_size, 'The number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', FLAGS.vector_size, 'The number of units in hidden layer 2')

if not FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_devidx)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

if os.path.isdir(FLAGS.output_dir) == False:
    os.mkdir(FLAGS.output_dir)

###########################################################
#
# Load EHR graphs
#
###########################################################

input_nodes, input_edges, input_stats = np.load(os.path.join(FLAGS.input_dir, "graph.npy"))
input_ctxpairs = np.load(os.path.join(FLAGS.input_dir, "ctxpairs.npy"))
input_seqdict, input_testindices = np.load(os.path.join(FLAGS.input_dir, "patients.npy"))

if os.path.isfile(FLAGS.label_file):
    input_nodelabels = np.load(FLAGS.label_file)
else:
    input_nodelabels = None

input_num_nodes, input_num_events, input_num_concepts = input_stats 
input_adjmat, input_degrees = preprocessing.normalize_graph(input_edges, input_num_nodes)

# We use identity features here, but any other node features can be utilized
input_feature = sp.identity(input_num_nodes) 
input_num_nodes, input_num_feats = input_feature.shape
input_feature = preprocessing.sparse_to_tuple(input_feature.tocoo())

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("1. Create minibatch iterators")
ti_minibatch = CtxPairMinibatchIterator(
    ctx_pairs = input_ctxpairs,
    batch_size = FLAGS.ti_batch_size,
    dropout = FLAGS.ti_dropout
)

tv_minibatch = SeqMinibatchIterator(
    sequences_dict = input_seqdict,
    labels_dict = None,
    test_indices = input_testindices,
    num_nodes = input_num_nodes,
    num_events = input_num_events,
    batch_size = FLAGS.tv_batch_size,
    dropout = FLAGS.tv_dropout
)

print("2. Defining placeholders")
placeholders = construct_placeholders(tv_minibatch.max_length, input_num_nodes)

print("3. Create models")
ti_model = TimeInvariantNode(
    placeholders = placeholders,
    adj_mat = input_adjmat,
    feature = input_feature,
    num_feats = input_num_feats,
    num_nodes = input_num_nodes
)

tv_model = TimeVariantNode(
    placeholders = placeholders,
    adj_mat = input_adjmat,
    feature = ti_model.first_embeds,
    degrees = input_degrees,
    num_steps = tv_minibatch.max_length,
    num_nodes = input_num_nodes
)

print("4. Create optimizers")
uni_opt = UnifiedOptimizer(
    ti_model = ti_model,
    tv_model = tv_model,
    ti_degrees = input_degrees,
    negsample_size = FLAGS.negsample_size,
    placeholders = placeholders,
    ti_batch_size = FLAGS.ti_batch_size,
    tv_batch_size = FLAGS.tv_batch_size
)

print("Initialize session")
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

###########################################################
#
# Train model
#
###########################################################

print("Train model")
itr = 0
k = FLAGS.recall_at
while itr <= FLAGS.n_iters:
    # Construct feed dictionary
    ti_feed_dict = ti_minibatch.next_minibatch_feed_dict(placeholders=placeholders)
    tv_feed_dict = tv_minibatch.next_minibatch_feed_dict(placeholders=placeholders)

    feed_dict = dict()
    feed_dict.update(ti_feed_dict)
    feed_dict.update(tv_feed_dict)

    # Training step: run single weight update
    outs = sess.run([uni_opt.opt_op, uni_opt.cost], feed_dict=feed_dict)
    train_cost = outs[1]

    if itr % FLAGS.n_printiters == 0:
        test_feed_dict, test_seqlens = tv_minibatch.test_minibatch_feed_dict(placeholders=placeholders)
        test_outs = sess.run([ti_model.ti_embeds, tv_model.inputs, uni_opt.tv_opt.softmax], feed_dict=test_feed_dict)
        ti_embeds, tv_labels, tv_outputs = test_outs

        recall = metrics.compute_recall_k(tv_labels[:,:,:input_num_events], tv_outputs[:,:,:input_num_events], test_seqlens, k)
        
        if input_nodelabels is not None:
            nmi = metrics.compute_clustering_nmi(ti_embeds[:input_num_events], input_nodelabels[:input_num_events])
            print("Minibatch-iter : %d\tMinibatch-cost : %f\tClustering NMI : %f\tRecall@d : %f" % (itr, train_cost, nmi, k, recall))
        else:
            print("Minibatch-iter : %d\tMinibatch-cost : %f\tRecall@%d : %f" % (itr, train_cost, k, recall))

        np.save(os.path.join(FLAGS.output_dir, "embedding_%d.npy"%itr), ti_embeds)

    itr += 1

print("Optimization finished!")

# Save the embedding vectors of events and concepts
np.save(os.path.join(FLAGS.output_dir, "embedding_final.npy"), ti_embeds)

# Save the trained entire model (including LSTM)
#saver.save(sess, "model/trained_model")
