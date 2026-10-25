import math

import torch
import torch.nn.functional as F


def entropy(p):
    return -(p * (p + 1e-10).log()).sum(dim=-1)


def _probs(logits_list, valid_mask):
    out = []
    for logits in logits_list:
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        out.append(F.softmax(logits, dim=-1))
    return out


def compute_jsd(logits_list, valid_mask=None, normalize=True):
    """Generalised Jensen-Shannon divergence between the heads' move distributions.

    normalize=True divides by its maximum, ln(min(#heads, #legal moves)), so the value is in
    [0, 1] and one threshold means the same thing in every position and game. (The old raw
    threshold 1.5 was above the maximum in most positions, so the JSD trigger never fired.)
    """
    probs = _probs(logits_list, valid_mask)
    avg = torch.stack(probs, dim=0).mean(dim=0)
    jsd = entropy(avg) - torch.stack([entropy(p) for p in probs], dim=0).mean(dim=0)
    if not normalize:
        return jsd
    n_valid = valid_mask.sum(-1).float() if valid_mask is not None else torch.full_like(jsd, avg.shape[-1])
    cap = torch.clamp(torch.minimum(n_valid, torch.full_like(n_valid, len(logits_list))), min=2.0)
    return (jsd / cap.log()).clamp(0.0, 1.0)


def get_max_prob(logits_list, valid_mask=None):
    avg = torch.stack(_probs(logits_list, valid_mask), dim=0).mean(dim=0)
    return avg.max(dim=-1).values


def should_plan(logits_list, valid_mask=None, jsd_threshold=0.5, max_prob_threshold=None):
    """Plan when normalised JSD >= jsd_threshold (or, if given, when max prob < max_prob_threshold).

    Get jsd_threshold from calibration (evaluate_v2.py), never by tuning on evaluation games.
    """
    jsd = compute_jsd(logits_list, valid_mask)
    max_prob = get_max_prob(logits_list, valid_mask)
    use = jsd >= jsd_threshold
    if max_prob_threshold is not None:
        use = use | (max_prob < max_prob_threshold)
    return use, jsd, max_prob


def max_jsd(num_heads: int, num_valid: int) -> float:
    return math.log(max(2, min(num_heads, num_valid)))
