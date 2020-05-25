import numpy as np
import scipy.sparse as sp

def normalize_graph(graph_edges, n_entities):
    mat_edges, data = np.array(graph_edges[0]), np.array(graph_edges[1])
    csr_adjmat = sp.csr_matrix((data, (mat_edges[:, 0], mat_edges[:, 1])), shape = (n_entities, n_entities))
    mat = sp.coo_matrix(csr_adjmat)

    if mat.shape[0] == mat.shape[1]:
        #mat = mat + sp.eye(mat.shape[0])
        rowsum = np.array(mat.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        mat_normalized = mat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    else:
        rowsum = np.array(mat.sum(1))
        colsum = np.array(mat.sum(0))
        rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        mat_normalized = rowdegree_mat_inv.dot(mat).dot(coldegree_mat_inv).tocoo()
    return sparse_to_tuple(mat_normalized), rowsum.squeeze()

def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
