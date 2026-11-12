import numpy as np, scipy.sparse as sp, torch

def build_adj(train, nU, nA):
    u, a = train['u'].values, train['a'].values + nU
    src, dst = np.concatenate([u, a]), np.concatenate([a, u])      # não direcionado
    order = np.argsort(src, kind='stable')
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=nU + nA))])
    return indptr, dst[order]

def random_walk_counts(indptr, indices, num_walks=50, walk_len=6, seed=0):
    n = len(indptr) - 1
    deg = np.diff(indptr)
    assert deg.min() > 0, 'há nó isolado no grafo de treino'
    rng = np.random.default_rng(seed)
    start = np.repeat(np.arange(n), num_walks)
    cur, rows, cols = start.copy(), [], []
    for _ in range(walk_len):
        r = (rng.random(len(cur)) * deg[cur]).astype(np.int64)
        cur = indices[indptr[cur] + r]                  # passo uniforme entre vizinhos
        rows.append(start); cols.append(cur)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    C = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    C = (C - sp.diags(C.diagonal())).tocsr(); C.eliminate_zeros()   # remove auto-visitas
    return C                                            # C[i, j] = visitas de j nas caminhadas de i

def walk_graph(C, T=20):
    '''Top-T mais visitados de cada nó. Aresta vizinho -> centro, peso = visitas normalizadas.'''
    src, dst, w = [], [], []
    for i in range(C.shape[0]):
        a, b = C.indptr[i], C.indptr[i + 1]
        cols, vals = C.indices[a:b], C.data[a:b]
        if len(vals) > T:
            top = np.argpartition(-vals, T)[:T]; cols, vals = cols[top], vals[top]
        src.append(cols); dst.append(np.full(len(cols), i)); w.append(vals / vals.sum())
    ei = torch.as_tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long)
    return ei, torch.as_tensor(np.concatenate(w), dtype=torch.float32)