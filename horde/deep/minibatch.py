from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class CtxPairMinibatchIterator(object):
    def __init__(self, ctx_pairs, batch_size=100, dropout=0.0):
        self.train_ctxpairs = ctx_pairs
        self.batch_size = batch_size
        self.dropout = dropout
        self.feed_dict = dict() 

        self.batch_iter = 0
        self.batch_maxiter = len(ctx_pairs) // self.batch_size 

    def is_end(self):
        finished = True if self.batch_iter == self.batch_maxiter else False
        return finished

    def _update_feed_dict(self, batch_ctx_pairs, placeholders):
        self.feed_dict.update({placeholders['batch_ti_pair']: batch_ctx_pairs})
        self.feed_dict.update({placeholders['ti_dropout']: self.dropout})

    def next_minibatch_feed_dict(self, placeholders):
        if self.is_end():
            self.shuffle()

        begin = self.batch_size * self.batch_iter
        end = begin + self.batch_size
        #end = begin + self.batch_size if self.batch_iter != self.batch_maxiter-1 else len(self.train_ctxpairs)
        batch_ctx_pairs = self.train_ctxpairs[begin:end]
        self._update_feed_dict(batch_ctx_pairs, placeholders)
        self.batch_iter += 1
        return self.feed_dict
  
    def shuffle(self):
        self.train_ctxpairs = np.random.permutation(self.train_ctxpairs)
        self.batch_iter = 0

class SeqMinibatchIterator(object):
    def __init__(self, sequences_dict, labels_dict, test_indices, num_nodes, num_events, batch_size=100, dropout=0.0, noise_size=10, noise_type="insertion"):
        self.num_nodes = num_nodes
        self.num_events = num_events
        self.batch_size = batch_size
        self.dropout = dropout
        self.feed_dict = dict()

        self.train_sequences, self.train_labels, self.test_sequences, self.test_labels = self._preprocess_sequences(sequences_dict, labels_dict, test_indices)
        self.train_seqlens = np.array([len(seq) for seq in self.train_sequences])
        self.test_seqlens = np.array([len(seq) for seq in self.test_sequences])
        self.max_length = max(max(self.train_seqlens), max(self.test_seqlens))

        self.train_seqinputs = self._build_multihot_vectors(self.train_sequences)
        self.test_seqinputs = self._build_multihot_vectors(self.test_sequences)

        self.noise_sequences, self.noise_tindices, self.noises = self._add_noise_sequences(sequences_dict, noise_size=noise_size, noise_type=noise_type)
        self.noise_seqinputs = self._build_multihot_vectors(self.noise_sequences)

        self.batch_iter = 0
        self.batch_maxiter = len(self.train_sequences) // self.batch_size

        #self.shuffle()

    def _preprocess_sequences(self, sequences_dict, labels_dict, test_indices):
        train_sequences, train_labels = [], []
        test_sequences, test_labels = [], []

        test_sequences = [sequences_dict[test_index] for test_index in test_indices]
        for index in sequences_dict:
            if len(sequences_dict[index]) > 1:
                if index not in test_indices:
                    train_sequences.append(sequences_dict[index])
                #else: #
                #    train_sequences.append(sequences_dict[index][:-2])

        if labels_dict is not None:
            test_labels = [labels_dict[test_index] for test_index in test_indices]
            for index in sequences_dict:
                if len(sequences_dict[index]) > 1:
                    if index not in test_indices:
                        train_labels.append(labels_dict[index])

        return np.array(train_sequences), np.array(train_labels), np.array(test_sequences), np.array(test_labels)

    def _add_noise_sequences(self, sequences_dict, noise_size, noise_type):
        noise_sequences, noise_tindices, noises = [], [], []
        while len(noises) < self.batch_size*10:
            noise_key = np.random.choice(list(sequences_dict.keys()), 1)[0]

            if noise_type == "insertion":
                cand_tindices = [t for t, visit in enumerate(sequences_dict[noise_key]) if len([j for j in visit if j >= self.num_events]) > 0]
                if len(cand_tindices) == 0: continue
                target_tindex = np.random.choice(cand_tindices, 1)[0]
                target_visit = sequences_dict[noise_key][target_tindex]

                noise = []
                while len(noise) < noise_size:
                    n = np.random.randint(self.num_events)
                    if n in target_visit:
                        continue
                    else:
                        target_visit.append(n)
                        noise.append(n)
                noises.append(noise)

            elif noise_type == "deletion":
                cand_tindices = [t for t, visit in enumerate(sequences_dict[noise_key]) if len([j for j in visit if j >= self.num_events]) > 0 and len([j for j in visit if j < self.num_events]) >= noise_size]
                if len(cand_tindices) == 0: continue
                target_tindex = np.random.choice(cand_tindices, 1)[0]
                target_visit = sequences_dict[noise_key][target_tindex]

                noise = np.random.choice([e for e in target_visit if e < self.num_events], noise_size, replace=False)
                for n in noise:
                    target_visit.remove(n)
                noises.append(noise)

            noise_tindices.append(target_tindex)
            noise_sequences.append(sequences_dict[noise_key])

        return np.array(noise_sequences), np.array(noise_tindices), np.array(noises)

    def _build_multihot_vectors(self, sequences):
        multihot_seqinputs, pad = [], [0] * self.num_nodes
        for sequence in sequences:
            multihot_seqinput = []
            for visit in sequence:
                multihot = [0] * self.num_nodes
                for index in visit:
                    multihot[index] += 1
                multihot_seqinput.append(multihot)
            for i in range(len(sequence), self.max_length):
                multihot_seqinput.append(pad)
            multihot_seqinputs.append(multihot_seqinput)
        return multihot_seqinputs

    def is_end(self):
        finished = True if self.batch_iter == self.batch_maxiter else False
        return finished

    def _update_feed_dict(self, batch_seqs, placeholders):
        self.feed_dict.update({placeholders['batch_tv_seq']: batch_seqs})
        self.feed_dict.update({placeholders['tv_dropout']: self.dropout})

    def next_minibatch_feed_dict(self, placeholders):
        if self.is_end():
            self.shuffle()

        begin = self.batch_size * self.batch_iter
        end = begin + self.batch_size
        batch_seqs = self.train_seqinputs[begin:end]
        self._update_feed_dict(batch_seqs, placeholders)
        self.batch_iter += 1
        return self.feed_dict
 
    def noise_minibatch_feed_dict(self, placeholders):
        self.feed_dict.update({placeholders['batch_tv_seq']: self.noise_seqinputs})
        self.feed_dict.update({placeholders['ti_dropout']: 0.0})
        self.feed_dict.update({placeholders['tv_dropout']: 0.0})
        return self.feed_dict

    def test_minibatch_feed_dict(self, placeholders):
        self.feed_dict.update({placeholders['batch_tv_seq']: self.test_seqinputs})
        self.feed_dict.update({placeholders['ti_dropout']: 0.0})
        self.feed_dict.update({placeholders['tv_dropout']: 0.0})
        return self.feed_dict, self.test_seqlens

    def shuffle(self):
        self.train_seqinputs = np.random.permutation(self.train_seqinputs)
        self.batch_iter = 0
