# Maintenance and execution scripts

The maintainer's lightweight tooling environment on Trillium is:

```bash
module load StdEnv/2023 python/3.12.4
source /home/jwj/venvs/kaggle-cli/bin/activate
```

These paths and module versions describe that account. On other systems, use your own environment
and credentials; the browser-based Kaggle starter is also available. The packaging script requires
only Python 3.10+ and its standard library.

| Script | Purpose |
| --- | --- |
| `kaggle.sh` | Load the above environment and call the CLI; credentials are read from HOME |
| `check_kaggle_quota.py` | Save quota using `total_seconds()` so SDK serialization cannot drop whole days |
| `download_competition.sh` | Download the complete ZIP and verify extraction against the inventory; run on a datamover |
| `extract_competition.py` | Check paths, names, sizes, and ZIP CRC; overwrites matching target input files |
| `audit_data.py` | Inspect task/graph schemas, NPZ headers, wheel metadata, and TGZ members on a datamover |
| `collect_resources.py` | Read official pages, public Code, discussion replies, and leaderboard CSV |
| `collect_discussion_bodies.py` | Read discussion opening posts, authors, and complete reply trees |
| `index_resources.py` | Generate English navigation and coverage records from local snapshots without executing upstream code |
| `render_notebook_reading.py` | Render English manual reviews with source hashes and line-range checks; no automatic claims of completed reading |
| `download_eval_assets.sh` / `verify_eval_assets.py` | Download and verify the required base and full official wheelhouse on a datamover |
| `build_submission.py` | Build a deterministic ZIP with path, extension, size, and CRC checks; official compilation is separate |
| `prepare_kaggle_upload.py` | Generate a private CPU packaging notebook without model inference |
| `prepare_kaggle_runtime.py` | Generate a private offline L4×4 single-task notebook |
| `prepare_eval_image.sh` / `prepare_eval_environment.sh` | Prepare the pinned image and separate evaluation environment, validated in one-task testing |
| `validate_agent.py` | Invoke the official compiler and restrictions; requires the full runtime dependencies |
| `prepare_eval_task.py` | Retain inference fields and remove reference/test patches |
| `prepare_eval_run.py` | Freeze the agent, task, and bundle hashes without overwriting an existing run |
| `summarize_eval_smoke.py` | Validate empty/gold controls and distinguish execution from issue resolution |
| `run_eval_container.sh` / `evaluate_agent.py` | Separate inference and verification within a compute allocation |
| `slurm_eval_smoke.sh` | Single-GPU generation and empty/gold/agent verification entry point |
| `prepare_eval_git.sh` | Add container Git from signed Ubuntu packages on a datamover; record hashes and diff checks |
| `submit_competition.py` | Recheck the tested ZIP; dry-run by default, upload with `--submit`, and retain a duplicate-prevention receipt |
| `check_submission_status.py` | Query existing submissions and L4 notebooks without uploading again |

Collectors cache files. Explicitly choose which snapshot to refresh; rerunning one command does not
refresh the entire collection. A complete update also requires separate competition-file, Models-page,
wheelhouse, and Code-structure inventories. Preserve dated evidence and inspect changes before updating conclusions.
