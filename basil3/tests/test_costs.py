import numpy as np
import torch

from basil3.agents import CitadelAgent, Gate
from basil3.multiverse import opening, universes
from basil3.nets import Citadel


def test_probes_and_search_calls_are_counted():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = Citadel(ch=16, repr_blocks=1, dyn_blocks=1, n_ricks=2).eval()
    g = opening(universes("val", 1)[0], 0)
    obs, legal = g.encode(), g.legal_mask()

    never = CitadelAgent(net, Gate("never"), sims=6, rng=np.random.default_rng(0))
    assert legal[never.act(obs, legal)]
    assert never.probe_calls == never.search_calls == never.planned == 0

    random = CitadelAgent(net, Gate("random", rate=0, rng=np.random.default_rng(1)),
                          sims=6, rng=np.random.default_rng(2))
    assert legal[random.act(obs, legal)]
    assert random.probe_calls == min(3, int(legal.sum())) * net.n_ricks
    assert random.search_calls == 0

    always = CitadelAgent(net, Gate("always"), sims=6, rng=np.random.default_rng(3))
    assert legal[always.act(obs, legal)]
    assert always.planned == 1
    assert always.probe_calls == 0
    assert always.search_calls == always.expansions * net.n_ricks
