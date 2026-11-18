'''
Projeto de IA (N2) - Recomendação musical por agregação de vizinhança aprendida
Integrantes: Arthur Jones Bicalho dos Santos 10420317;
             Luis Felipe Santos do Nascimento 10420572
Síntese: GraphSAGE sobre a vizinhança das caminhadas, com pesos de agregação fixos ou aprendidos.
Histórico:
2026-11-10 | Arthur | Criação do módulo
2026-11-18 | Luis Felipe | Atualização da lista de integrantes
'''
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class WalkSAGEConv(MessagePassing):
    def __init__(self, d_in, d_out, learn_weights=True):
        super().__init__(aggr='add')
        self.learn_weights = learn_weights
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nbr = nn.Linear(d_in, d_out, bias=False)
        if learn_weights:
            self.score = nn.Sequential(nn.Linear(2 * d_in + 1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_w):
        src, dst = edge_index                              # vizinho -> centro
        if self.learn_weights:
            s = self.score(torch.cat([x[dst], x[src], edge_w.unsqueeze(-1)], dim=-1)).squeeze(-1)
            alpha = softmax(s, dst, num_nodes=x.size(0))   # pesos aprendidos, soma 1 por centro
        else:
            alpha = edge_w                                 # contagens normalizadas (fixas)
        agg = self.propagate(edge_index, x=self.lin_nbr(x), alpha=alpha, size=(x.size(0), x.size(0)))
        return self.lin_self(x) + agg

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

class RecGNN(nn.Module):
    def __init__(self, nU, nA, country, n_country, d=64, layers=2, learn_weights=True):
        super().__init__()
        self.nU = nU
        self.emb = nn.Embedding(nU + nA, d)
        self.cty = nn.Embedding(n_country + 1, d)           # índice n_country = artista (neutro)
        c = torch.cat([torch.as_tensor(country), torch.full((nA,), n_country)])
        self.register_buffer('c_idx', c.long())
        self.convs = nn.ModuleList([WalkSAGEConv(d, d, learn_weights) for _ in range(layers)])

    def forward(self, edge_index, edge_w):
        h = self.emb.weight + self.cty(self.c_idx)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_w)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h[:self.nU], h[self.nU:]                      # embeddings de usuários e de artistas