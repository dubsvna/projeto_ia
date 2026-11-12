import time, copy, numpy as np, pandas as pd, torch, torch.nn.functional as F
from src.model import RecGNN
from src.metrics import evaluate

@torch.no_grad()
def gnn_fn(model, ei, ew):
    model.eval(); dev = next(model.parameters()).device; ei, ew = ei.to(dev), ew.to(dev)
    zu, za = model(ei, ew)
    return lambda us: (zu[torch.as_tensor(us, device=zu.device)] @ za.T).cpu().numpy()

def train_gnn(D, ei, ew, learn_weights=True, seed=0, d=64, lr=1e-3, wd=1e-6, bs=65536,
              max_epochs=100, eval_every=5, patience=3):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecGNN(D['nU'], D['nA'], D['country'], D['n_country'], d, 2, learn_weights).to(dev)
    ei, ew = ei.to(dev), ew.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pu = torch.as_tensor(D['train']['u'].values, device=dev)
    pa = torch.as_tensor(D['train']['a'].values, device=dev)
    best, bad, best_state, hist = -1.0, 0, None, []
    t0 = time.perf_counter()
    for ep in range(1, max_epochs + 1):
        model.train(); losses = []
        perm = torch.randperm(len(pu), device=dev)
        for s in range(0, len(pu), bs):
            idx = perm[s:s + bs]
            zu, za = model(ei, ew)
            u, p = zu[pu[idx]], za[pa[idx]]
            n = za[torch.randint(0, D['nA'], (len(idx),), device=dev)]   # negativos amostrados
            loss = -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        rec = {'epoch': ep, 'loss': float(np.mean(losses))}
        if ep % eval_every == 0:
            summ, _, _ = evaluate(gnn_fn(model, ei, ew), D['train_csr'], D['val_csr'], Ks=(10,))
            rec['val_ndcg@10'] = summ['ndcg@10']
            if rec['val_ndcg@10'] > best:
                best, bad, best_state = rec['val_ndcg@10'], 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
        hist.append(rec)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), time.perf_counter() - t0