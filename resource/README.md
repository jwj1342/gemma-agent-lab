# Research resource index

Collection snapshot: October 4–5, 2026 (UTC). Source material and our interpretations are stored separately.
Read the [research report](../docs/competition-research.md) and
[coverage record](manifests/research-collection.json) for scope and limitations.

| Location | Contents and coverage |
| --- | --- |
| [competition/](competition/pages.json) | Ten complete official pages, including Overview, Data description, rules, dates, and L4 guidance |
| [notebooks/](notebooks/README.md) | All 112 listed public notebooks downloaded; one has empty source |
| [Notebook reading notes](notebooks/reading-notes.md) | Per-entry execution paths, reusable ideas, evidence limits, and source locations |
| [Official starter](notebooks/official/getting-started-gemma-4-developer-agent.ipynb) | Requested official walkthrough, reviewed statically |
| [discussions/](discussions/README.md) | 116 topics and 370 opening posts/replies, with authors and timestamps |
| [models/](models/README.md) | All four listed instances and available cards; required Gemma v2 weights were downloaded separately |
| [leaderboard/](leaderboard/README.md) | Official CSV and distribution across 1,755 scored teams |
| [manifests/](manifests/data-audit.json) | File inventories, verification, structural audits, and storage snapshots |
| [harness-source/](harness-source/README.md) | Python source and metadata from three official wheels, for static inspection |
| `downloads/` | Local dataset ZIP, download caches, and harness wheels; excluded from Git |

Third-party sources retain their original content, language, and attribution. Do not directly execute
notebook installation/cleanup cells or treat reported scores as reproduced results. English navigation
and reading notes are maintained by the team. External links, image attachments, and every line of
third-party source have not all been audited; no third-party training or evaluation notebook was executed.
Our measured results are in the [H100](manifests/hpc-smoke.json), [L4](manifests/kaggle-runtime-smoke.json),
and [official score](manifests/competition-result.json) records.

To collect pages, Code, and reply trees, use `scripts/collect_resources.py`; to fill discussion
opening posts and authors, use `scripts/collect_discussion_bodies.py`. After reviewing a new snapshot,
run `scripts/index_resources.py` to regenerate navigation. Models-page and wheelhouse inventories
were collected separately and are not automatically refreshed by the base collector.
The collection tools use the maintainer's HOME Kaggle CLI environment.
