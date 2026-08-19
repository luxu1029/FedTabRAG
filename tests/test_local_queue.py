import torch

from src.federated.local_queue import audit, empty_queue, push


def test_queue_is_bounded_fifo_and_refreshes_duplicates():
    q = empty_queue()
    for i in range(4):
        x = torch.tensor([float(i)], requires_grad=True)
        push(q, f"e{i}", x, x, client_id=2, capacity=3)
    assert q["ids"] == ["e1", "e2", "e3"]
    push(q, "e2", torch.tensor([9.0], requires_grad=True),
         torch.tensor([9.0], requires_grad=True), 2, 3)
    assert q["ids"] == ["e1", "e3", "e2"]
    assert q["student"][-1].item() == 9


def test_queue_detaches_and_boundary_audit_is_clean():
    q = empty_queue()
    x = torch.tensor([1.0], requires_grad=True)
    push(q, "train:c:q::row::1", x, x, client_id=1, capacity=8)
    assert audit(q, 1) == {"cross_client": 0, "non_train": 0, "attached": 0}
