# FedTabRAG

FedTabRAG is an experimental research codebase for structure-aware federated
retrieval-augmented generation over financial tables. It contains the data
processing, table-evidence construction, retrieval evaluation, federated
training, ablation, and diagnostic scripts used in the project experiments.

## Repository contents

- `src/data/`: FinQA validation and table rendering utilities
- `src/table/`: hierarchical evidence construction and evidence mapping
- `src/retrieval/`: indexing, retrieval evaluation, and hard-negative mining
- `src/federated/`: federated training, aggregation, losses, and evaluation
- `src/evaluation/`: result merging and Flat/UniTable diagnostics
- `scripts/`: experiment orchestration and result-freezing scripts
- `configs/`: data, retrieval, QA, and federated experiment configurations
- `tests/`: project unit tests
- `WORKLOG.md`: verified experiment status and recommended continuation point

Large datasets, model weights, indexes, checkpoints, generated outputs, and
vendored third-party repositories are intentionally excluded from Git. See the
experiment documents and `WORKLOG.md` for artifact locations and protocol
details in the original workspace.

## Tests

The verified local environment is named `fedrag-test`:

```bash
env PYTHONPATH=. conda run -n fedrag-test pytest -q tests
```

The current project test suite contains 22 tests. Restrict collection to
`tests/`; vendored research repositories have separate dependencies and test
layouts.

## Current status

The Day 1–10 experiments are complete and frozen. The follow-up Flat versus
UniTable diagnostic tasks 1 and 2 are complete; task 3 has exported 90 manual
cases but still requires human annotation. Refer to `WORKLOG.md` before
continuing experiments.

## Data and third-party dependencies

FinQA data, BGE weights, UniTable, and FedE4RAG are not redistributed in this
repository. Obtain each dependency from its official source and comply with
its license and data-use terms. Paths expected by the scripts are documented
in the configuration files and experiment documents.

