"""Losses for Day-6 hierarchical retrieval; intentionally contains no KD."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LEVELS = ("table", "row", "cell")


def multi_positive_contrastive(logits, positive_mask, valid_query_mask=None):
    """InfoNCE with a set-valued numerator, so positives never oppose each other."""
    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have identical shapes")
    positive_mask = positive_mask.bool()
    valid = positive_mask.any(1)
    if valid_query_mask is not None:
        valid &= valid_query_mask.bool()
    if not valid.any():
        return logits.sum() * 0.0, 0
    z = logits[valid]
    m = positive_mask[valid]
    numerator = torch.logsumexp(z.masked_fill(~m, -torch.inf), dim=1)
    denominator = torch.logsumexp(z, dim=1)
    return (denominator - numerator).mean(), int(valid.sum().item())


def hierarchical_loss(level_logits, level_positive_masks, weights=None):
    """Weighted level losses with weights renormalized over available levels."""
    weights = weights or {"table": 0.2, "row": 0.3, "cell": 0.5}
    losses, counts = {}, {}
    active = []
    for level in LEVELS:
        loss, count = multi_positive_contrastive(
            level_logits[level], level_positive_masks[level])
        losses[level] = loss
        counts[level] = count
        if count:
            active.append(level)
    if not active:
        raise ValueError("batch has no valid positives at any level")
    normalizer = sum(weights[level] for level in active)
    total = sum(losses[level] * weights[level] / normalizer for level in active)
    return total, losses, counts


def margin_hard_negative_loss(query, positive, negative, margin=0.2):
    if not (len(query) == len(positive) == len(negative)):
        raise ValueError("query/positive/negative batch sizes differ")
    if not len(query):
        return query.sum() * 0.0
    pos = F.cosine_similarity(query, positive)
    neg = F.cosine_similarity(query, negative)
    return F.relu(float(margin) + neg - pos).mean()


def distillation_kl(student_logits, teacher_logits, valid_query_mask, temperature=2.0):
    """KL(teacher || student) with temperature scaling and query masking."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student logits must have identical shapes")
    valid = valid_query_mask.bool()
    if not valid.any():
        return student_logits.sum() * 0.0, 0
    s = student_logits[valid] / float(temperature)
    t = teacher_logits[valid].detach() / float(temperature)
    teacher_prob = F.softmax(t, dim=1)
    loss = F.kl_div(F.log_softmax(s, dim=1), teacher_prob,
                    reduction="batchmean") * float(temperature) ** 2
    return loss, int(valid.sum().item())


def hierarchical_kd(student_logits, teacher_logits, positive_masks, weights=None,
                    temperature=2.0):
    weights = weights or {"table": 0.2, "row": 0.3, "cell": 0.5}
    losses, counts, active = {}, {}, []
    for level in LEVELS:
        loss, count = distillation_kl(
            student_logits[level], teacher_logits[level],
            positive_masks[level].any(dim=1), temperature)
        losses[level], counts[level] = loss, count
        if count: active.append(level)
    if not active:
        raise ValueError("batch has no valid KD level")
    z = sum(weights[x] for x in active)
    total = sum(losses[x] * weights[x] / z for x in active)
    return total, losses, counts


def kd_weight_for_round(round_no, target=0.5, start_round=2, warmup_rounds=2):
    if round_no < start_round:
        return 0.0
    progress = min(1.0, (round_no - start_round + 1) / float(warmup_rounds))
    return float(target) * progress
