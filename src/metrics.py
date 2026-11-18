'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: avaliação comum a todos os modelos (Precision, Recall, NDCG, cobertura e popularidade @K).
Histórico:
2026-11-10 | Arthur | Criação do módulo
2026-11-18 | Luis Felipe | Atualização da lista de integrantes
'''
import numpy as np

def evaluate(score_fn, mask, target, Ks=(10, 50), batch=2000):
    nU, nA = target.shape
    Kmax = max(Ks)
    disc = 1.0 / np.log2(np.arange(2, Kmax + 2)); cdisc = np.cumsum(disc)
    pop = np.log1p(np.asarray(mask.sum(0)).ravel())            # popularidade (log) no treino
    users = np.where(target.getnnz(axis=1) > 0)[0]
    out = {K: {k: [] for k in ('prec', 'rec', 'ndcg', 'pop')} for K in Ks}
    rec_counts = {K: np.zeros(nA, dtype=np.int64) for K in Ks}
    for s in range(0, len(users), batch):
        us = users[s:s + batch]
        sc = np.array(score_fn(us), dtype=np.float32)          # cópia
        m = mask[us].tocoo()
        sc[m.row, m.col] = -np.inf                              # não recomendar o já visto
        top = np.argpartition(-sc, Kmax, axis=1)[:, :Kmax]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, 1), axis=1), 1)
        rel = target[us].toarray() > 0
        hits = np.take_along_axis(rel, top, 1)
        n_rel = rel.sum(1)
        for K in Ks:
            h = hits[:, :K]; nh = h.sum(1)
            out[K]['prec'].append(nh / K)
            out[K]['rec'].append(nh / n_rel)
            out[K]['ndcg'].append((h * disc[:K]).sum(1) / cdisc[np.minimum(n_rel, K) - 1])
            out[K]['pop'].append(pop[top[:, :K]].mean(1))
            rec_counts[K] += np.bincount(top[:, :K].ravel(), minlength=nA)
    summary, per_user = {}, {}
    for K in Ks:
        for k in ('prec', 'rec', 'ndcg', 'pop'):
            v = np.concatenate(out[K][k])
            per_user[f'{k}@{K}'] = v; summary[f'{k}@{K}'] = v.mean()
        summary[f'cov@{K}'] = (rec_counts[K] > 0).mean()        # cobertura de catálogo
    return summary, per_user, rec_counts