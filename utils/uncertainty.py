import torch
import torch.nn.functional as F


def compute_jsd(logits_list, valid_mask=None):
    probs_list = []
    for logits in logits_list:
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        probs_list.append(F.softmax(logits, dim=-1))

    probs_stack = torch.stack(probs_list, dim=0)
    avg_probs = probs_stack.mean(dim=0)

    avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum(dim=-1)
    mean_head_entropy = 0.0
    for p in probs_list:
        mean_head_entropy = mean_head_entropy + (-(p * (p + 1e-10).log()).sum(dim=-1))
    mean_head_entropy = mean_head_entropy / len(probs_list)

    jsd = avg_entropy - mean_head_entropy
    return jsd


def should_plan(logits_list, valid_mask=None, jsd_threshold=1.0, max_prob_threshold=0.6):
    jsd = compute_jsd(logits_list, valid_mask)

    probs_list = []
    for logits in logits_list:
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float('-inf'))
        probs_list.append(F.softmax(logits, dim=-1))
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    max_prob = avg_probs.max(dim=-1).values

    use_planning = (jsd >= jsd_threshold) | (max_prob < max_prob_threshold)
    return use_planning, jsd, max_prob
