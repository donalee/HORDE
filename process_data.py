import sys, os
import math
import numpy as np

input_file = sys.argv[1]
output_dir = sys.argv[2]
num_tests = int(sys.argv[3])

if !os.path.isdir(output_dir): os.mkdir(output_dir)

data = np.load(input_file)

# Identifying concept-concept edges by NPMI values

word_count, pair_count = {}, {}
window_size = 5
window_num = 0
for idx in range(len(data)):
    windows = []
    concepts = data[idx][3]

    if len(concepts) < window_size:
        windows = [concepts]
    else:
        windows = [concepts[j:j+window_size] for j in range(len(concepts) - window_size + 1)]

    window_num += len(windows)

    for window in windows:
        appeared = set()
        for concept in window:
            if concept in appeared: continue
            word_count[concept] = word_count.get(concept, 0) + 1
            appeared.add(concept)

    for window in windows:
        appeared = set()
        for i in range(1, len(window)):
            for j in range(i):
                if window[i] == window[j]: continue
                if (window[i], window[j]) in appeared or (window[j], window[i]) in appeared: continue
                pair_count[(window[i], window[j])] = pair_count.get((window[i], window[j]), 0) + 1
                pair_count[(window[j], window[i])] = pair_count.get((window[j], window[i]), 0) + 1
                appeared.add((window[i], window[j]))

concept_edges = [[], []]
for pair in pair_count:
    concept_i, concept_j = pair
    concept_ij_count = pair_count[pair]
    concept_i_count = word_count[concept_i]
    concept_j_count = word_count[concept_j]
    pmi = math.log((1.0 * concept_ij_count / window_num) / (1.0 * concept_i_count * concept_j_count / (window_num * window_num)))
    if pmi <= 0: continue
    npmi = -1.0 * pmi / math.log(concept_ij_count / window_num)
    concept_edges[0].append([concept_i, concept_j])
    concept_edges[1].append(npmi)

# Constructing an EHR graph

entity2id = {}
num_events, num_concepts = 0, 0
nodes = []

for idx in range(len(data)):
    events = data[idx][2]
    if len(events) == 0:
        continue
    for event in events:
        entity = "E:" + event
        if entity not in entity2id:
            entity2id[entity] = num_events
            nodes.append(entity)
            num_events += 1

for idx in range(len(data)):
    concepts = data[idx][3]
    if len(concepts) == 0:
        continue
    for concept in concepts:
        entity = "C:" + concept
        if entity not in entity2id:
            entity2id[entity] = num_events + num_concepts
            nodes.append(entity)
            num_concepts += 1

num_nodes = num_events + num_concepts

concept_concept_edges = [[entity2id["C:"+first], entity2id["C:"+second]] for first, second in concept_edges[0]]
concept_concept_weights = [npmi for npmi in concept_edges[1]]

identity_edges = [[i, i] for i in range(len(nodes))]
identity_weights = [1] * len(nodes)

edges = identity_edges + concept_concept_edges
weights = identity_weights + concept_concept_weights

np.save(os.path.join(output_dir, "graph.npy"), [nodes, [edges, weights], [num_nodes, num_events, num_concepts]])

# Building entity-entity context-pairs
# Building patients' visit sequences

ctx_pairs = []
visit_seqs = {}
for idx in range(len(data)):
    pid, vid, events, concepts = data[idx]

    events = [entity2id["E:" + event] for event in events]
    concepts = [entity2id["C:" + concept] for concept in concepts]
    visit = events + list(set(concepts))
  
    if len(visit) == 0: continue
    visit_seqs[pid] = visit_seqs.get(pid, []) + [visit]

    for src in visit:
        for dst in visit:
            if src == dst: continue
            ctx_pairs.append((src, dst))

cand_pids = [p for p in visit_seqs if len(visit_seqs[p]) > 1]
test_pids = np.random.choice(cand_pids, num_tests, replace=False)

np.save(os.path.join(output_dir, "ctxpairs.npy"), ctx_pairs)
np.save(os.path.join(output_dir, "patients.npy"), [visit_seqs, test_pids])
