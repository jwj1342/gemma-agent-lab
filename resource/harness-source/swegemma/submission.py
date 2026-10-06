"""Submission packaging and evaluation path discovery for SWE-gemma."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd


class ParticipantVisibleError(Exception):
    """Errors raised here will be displayed to participants in the Kaggle UI."""

    pass


def get_eval_root() -> Path:
    """Locate the root evaluation directory containing competition data."""
    if 'KAGGLE_EVAL_ROOT' in os.environ:
        return Path(os.environ['KAGGLE_EVAL_ROOT'])
    for candidate in [
        Path('/kaggle/input/competitions/gemma-4-developer-agent'),
        Path('/kaggle/input/gemma-4-developer-agent'),
        Path('/kaggle/input/datasets/metric/swegemma-evaluation'),
        Path('data'),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        'Evaluation root not found. Set KAGGLE_EVAL_ROOT or verify Kaggle dataset mount.'
    )


def generate_standard_submission(submission_dir: str) -> None:
    """Processes an extracted agent submission archive to produce a standard submission file."""
    possible_dirs: list[str] = []
    for d in [submission_dir, '/kaggle/working', '/kaggle/tmp']:
        if d and d not in possible_dirs:
            possible_dirs.append(d)

    agent_yamls: list[str] = []
    for search_dir in possible_dirs:
        if os.path.exists(search_dir):
            dir_candidates: list[str] = []
            for pattern in ('**/agent.yaml', '**/root_agent.yaml'):
                dir_candidates.extend(
                    glob.glob(os.path.join(search_dir, pattern), recursive=True)
                )
            if dir_candidates:
                dir_candidates.sort(
                    key=lambda p: (
                        len(Path(p).parts),
                        0 if Path(p).name == 'agent.yaml' else 1,
                        p,
                    )
                )
                agent_yamls.extend(dir_candidates)

    if not agent_yamls:
        found_info = '\n'.join(
            f'{d}: {os.listdir(d) if os.path.exists(d) else "not found"}'
            for d in possible_dirs
        )
        raise ParticipantVisibleError(
            f'No agent.yaml or root_agent.yaml found in submission. Searched directories:\n{found_info}'
        )

    agent_dir = os.path.dirname(agent_yamls[0])
    try:
        eval_root = get_eval_root()
        sample_sub_candidates = [
            eval_root / 'sample_submission.parquet',
            eval_root / 'sample_submission.csv',
        ]
    except Exception:
        sample_sub_candidates = [
            Path('sample_submission.parquet'),
            Path('sample_submission.csv'),
        ]

    sample_sub_path = next((p for p in sample_sub_candidates if p.exists()), None)
    if sample_sub_path is None:
        # Generate default two-row sample submission if not present
        sample_sub_df = pd.DataFrame({'id': ['public', 'private']})
    elif sample_sub_path.suffix == '.parquet':
        sample_sub_df = pd.read_parquet(sample_sub_path)
    else:
        sample_sub_df = pd.read_csv(sample_sub_path)

    row_id_col = str(sample_sub_df.columns[0])

    predictions = []
    for item in sample_sub_df.itertuples(index=False):
        predictions.append(
            {
                row_id_col: getattr(item, row_id_col),
                'prediction': agent_dir,
            }
        )

    submission_df = pd.DataFrame(predictions)
    submission_df.to_parquet('submission.parquet', index=False)
    print(f'Standard submission generated with agent_dir: {agent_dir}')
