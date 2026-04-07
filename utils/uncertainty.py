import torch
import torch.nn.functional as F


def entropy(p):
    return -(p * (p + 1e-10).log()).sum(dim=-1)


def compute_jsd(logits_list, valid_mask=None):
    probs_list = []
    for logits in logits_list:
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        probs_list.append(F.softmax(logits, dim=-1))

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    h_avg = entropy(avg_probs)
    h_each = torch.stack([entropy(p) for p in probs_list], dim=0).mean(dim=0)
    return h_avg - h_each


def get_max_prob(logits_list, valid_mask=None):
    probs_list = []
    for logits in logits_list:
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        probs_list.append(F.softmax(logits, dim=-1))
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs.max(dim=-1).values


def should_plan(logits_list, valid_mask=None, jsd_threshold=0.5, max_prob_threshold=0.6):
    jsd = compute_jsd(logits_list, valid_mask)
    max_prob = get_max_prob(logits_list, valid_mask)
    use_planning = (jsd >= jsd_threshold) | (max_prob < max_prob_threshold)
    return use_planning, jsd, max_prob
