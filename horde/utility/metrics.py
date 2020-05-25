import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import heapq

def compute_clustering_nmi(embeds, labels):
    n_clusters = 257 # MIMIC : 281, CDW : 257
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embeds)
    nmi = normalized_mutual_info_score(kmeans_labels, labels, 'arithmetic')
    return nmi

def compute_recall_k(labels, outputs, seqlens, k=10):
    recall, num = 0, 0
    for i in range(len(seqlens)):
        for t in range(1, seqlens[i]):
            num += 1

            target_labels = labels[i,t,:]
            target_outputs = outputs[i,t-1,:]
            topk_entities = heapq.nlargest(k, range(len(target_outputs)), target_outputs.take)

            if sum(target_labels) != 0:
                recall += sum(target_labels[topk_entities])/sum(target_labels)
            else:
                recall += 1.0
    return recall/num

def plot_lowdim_space(embeds, num_events, itr, output_dir='plots'):
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    low_embeds = TSNE(n_components=2, metric='cosine').fit(embeds)
    low_embeds = low_embeds.embedding_

    plt.clf()
    plt.scatter(low_embeds[:num_events, 0], low_embeds[:num_events, 1], c='blue', alpha=0.2)
    plt.scatter(low_embeds[num_events:, 0], low_embeds[num_events:, 1], c='red', alpha=0.2)
    plt.savefig(os.path.join(output_dir, "embedding_space_%d.png"% itr))

