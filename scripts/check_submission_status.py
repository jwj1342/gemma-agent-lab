"""Read existing competition/notebook status; never submit, restart, or cancel."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def plain(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, 'name'):
        return value.name
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-ref', type=int, required=True)
    parser.add_argument('--kernel', default='jwj1342/gemma4-official-runtime-smoke')
    parser.add_argument('--output', type=Path,
                        default=Path('resource/manifests/kaggle-evaluation-status.json'))
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    # The extended API already unwraps the list_submissions response.
    submissions = api.competition_submissions('gemma-4-developer-agent', page_size=100)
    submission = next((s for s in submissions or [] if s.ref == args.submission_ref), None)
    if submission is None:
        raise RuntimeError('Submission absent from latest 100 entries; inspect history without resubmitting')
    record = {key: plain(getattr(submission, key, None)) for key in
              ['ref', 'file_name', 'date', 'description', 'status', 'public_score',
               'private_score', 'error_description', 'total_bytes']}
    kernel = api.kernels_status(args.kernel)
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'competition': 'gemma-4-developer-agent', 'submission': record,
              'private_runtime_notebook': {'ref': args.kernel, 'status': plain(kernel.status),
                                          'failure_message': kernel.failure_message},
              'scope': 'Authenticated remote status snapshot. A notebook COMPLETE status requires output-log verification; official public score is distinct from private/final score.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
