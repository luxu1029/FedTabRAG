"""Bounded client-local detached embedding queue."""


def empty_queue():
    return {"ids": [], "student": [], "teacher": [], "split": [], "client": []}


def push(queue, evidence_id, student_embedding, teacher_embedding, client_id, capacity):
    if evidence_id in queue["ids"]:
        index = queue["ids"].index(evidence_id)
        for key in queue:
            queue[key].pop(index)
    queue["ids"].append(evidence_id)
    queue["student"].append(student_embedding.detach())
    queue["teacher"].append(teacher_embedding.detach())
    queue["split"].append("train")
    queue["client"].append(client_id)
    while len(queue["ids"]) > capacity:
        for key in queue:
            queue[key].pop(0)


def audit(queue, expected_client):
    return {
        "cross_client": sum(x != expected_client for x in queue["client"]),
        "non_train": sum(x != "train" for x in queue["split"]),
        "attached": sum(x.requires_grad for key in ("student", "teacher")
                        for x in queue[key]),
    }
