import torch

from src.federated.fedtabrag_losses import (
    distillation_kl, hierarchical_loss, hierarchical_kd, kd_weight_for_round,
    margin_hard_negative_loss, multi_positive_contrastive)


def test_multi_positive_not_treated_as_negative():
    logits = torch.tensor([[3.0, 2.0, -1.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss, count = multi_positive_contrastive(logits, mask)
    loss.backward()
    assert count == 1
    assert logits.grad[0, 0] <= 0 and logits.grad[0, 1] <= 0
    assert logits.grad[0, 2] >= 0


def test_missing_levels_are_masked_and_weights_renormalized():
    logits = {x: torch.tensor([[2.0, 0.0]], requires_grad=True)
              for x in ("table", "row", "cell")}
    masks = {
        "table": torch.tensor([[True, False]]),
        "row": torch.tensor([[False, False]]),
        "cell": torch.tensor([[True, False]]),
    }
    total, losses, counts = hierarchical_loss(logits, masks)
    expected = losses["table"] * (0.2 / 0.7) + losses["cell"] * (0.5 / 0.7)
    assert torch.allclose(total, expected)
    assert counts == {"table": 1, "row": 0, "cell": 1}


def test_hard_margin_zero_when_separated_and_positive_when_violated():
    q = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([[1.0, 0.0]])
    easy = torch.tensor([[-1.0, 0.0]])
    hard = torch.tensor([[0.9, 0.1]])
    assert margin_hard_negative_loss(q, pos, easy).item() == 0
    assert margin_hard_negative_loss(q, pos, hard).item() > 0


def test_kd_identical_near_zero_and_perturbation_increases():
    teacher = torch.tensor([[2.0, 0.5, -1.0]])
    same = teacher.clone().requires_grad_()
    zero, _ = distillation_kl(same, teacher, torch.tensor([True]), 2.0)
    perturbed = torch.tensor([[-1.0, 0.5, 2.0]], requires_grad=True)
    changed, _ = distillation_kl(perturbed, teacher, torch.tensor([True]), 2.0)
    assert abs(zero.item()) < 1e-6
    assert changed.item() > zero.item() + 0.1


def test_kd_invariant_to_common_logit_shift():
    s = torch.tensor([[0.2, 0.7, -0.1]])
    t = torch.tensor([[0.9, -0.2, 0.4]])
    a, _ = distillation_kl(s, t, torch.tensor([True]), 2.0)
    b, _ = distillation_kl(s + 17, t + 17, torch.tensor([True]), 2.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_kd_teacher_detached_student_has_gradient_and_missing_mask():
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    student = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    loss, count = distillation_kl(student, teacher, torch.tensor([True, False]), 2.0)
    loss.backward()
    assert count == 1 and teacher.grad is None
    assert student.grad is not None and student.grad[0].abs().sum() > 0
    assert student.grad[1].abs().sum() == 0


def test_kd_warmup_schedule():
    assert kd_weight_for_round(1) == 0
    assert kd_weight_for_round(2) == 0.25
    assert kd_weight_for_round(3) == 0.5
