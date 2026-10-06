# Experiment artifacts

Use a separate `<date>-<short-name>/` directory for each experiment. Record Git revision,
module/dependency versions, model/data versions, seeds, resources, metrics, logs, and recovery paths.
Artifacts are excluded from Git. Existing runs cover collection, packaging, container preparation,
and evaluation; they do not include completed training.

- `hpc-smoke-20261005-r4/`: successful offline single-task evaluation, with official compilation,
  empty/gold controls, generated patch, and independent tests. Earlier attempts retain debugging evidence.
- `pipeline-baseline/`: reproducible ZIP and completed Kaggle CPU packaging output.
- `kaggle-runtime-smoke/`: completed private L4×4 verification and downloaded evidence.
- `slides-preview/`: presentation build logs and rendered review images.

Small summaries are tracked in `resource/manifests/hpc-smoke.json`,
`kaggle-runtime-smoke.json`, and `kaggle-evaluation-status.json`.
Submission receipts remain in their original runs; copying a run is not a reason to submit again.

On Trillium, writable runtime caches belong on SCRATCH. Back up important source and final results separately;
this directory is not backed up by the cluster. These historical runs are not included in a clone.
