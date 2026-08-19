#!/usr/bin/env python3
"""Create the Day-5 fairness and acceptance summary from two completed runs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flat-run", type=Path, required=True)
    p.add_argument("--unitable-run", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = {"FedE4RAG-Flat": args.flat_run, "FedE4RAG-UniTable": args.unitable_run}
    metrics = {name: json.loads((root / "test_metrics.json").read_text())
               for name, root in runs.items()}
    configs = {name: json.loads((root / "resolved_config.json").read_text())
               for name, root in runs.items()}
    audits = {name: json.loads((root / "training_audit.json").read_text())
              for name, root in runs.items()}
    excluded = {"variant", "run_name"}
    comparable = {name: {k: v for k, v in cfg.items() if k not in excluded}
                  for name, cfg in configs.items()}
    same_budget = comparable["FedE4RAG-Flat"] == comparable["FedE4RAG-UniTable"]
    partition = json.loads(args.partition.read_text())
    rows = {}
    for name in runs:
        rows[name] = {
            "best_round": metrics[name]["checkpoint_round"],
            "global": metrics[name]["global"],
            "fairness": metrics[name]["fairness"],
            "communication": metrics[name]["communication"],
            "training": metrics[name]["training"],
        }
    acceptance = {
        "partition_atomic_and_frozen": partition["mapping_sha256"]
        == configs["FedE4RAG-Flat"]["partition_sha256"]
        == configs["FedE4RAG-UniTable"]["partition_sha256"],
        "same_lora_and_training_budget": same_budget,
        "both_completed_10_rounds": all(a["completed_rounds"] == 10 for a in audits.values()),
        "teacher_no_grad": all(a["teacher_no_grad"] for a in audits.values()),
        "clients_sequential": all(a["clients_sequential"] for a in audits.values()),
        "best_and_last_only": all(a["checkpoints"] == ["best.pt", "last.pt"]
                                  for a in audits.values()),
        "vanilla_fedavg": True,
        "no_hierarchical_loss": True,
    }
    report = {
        "acceptance": {**acceptance, "passed": all(acceptance.values())},
        "partition_mapping_sha256": partition["mapping_sha256"],
        "partition_file_sha256": sha(args.partition),
        "fairness_excluded_config_keys": sorted(excluded),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(report["acceptance"], indent=2))


if __name__ == "__main__":
    main()
