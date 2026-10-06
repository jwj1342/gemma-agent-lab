"""Check a tested run; with --submit, upload its exact ZIP once to Kaggle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from verify_eval_run import verify_inputs
from summarize_eval_smoke import summarize


COMPETITION = 'gemma-4-developer-agent'


def check_run(run):
    manifest = verify_inputs(run)
    result = summarize(run)
    if not result.get('pipeline_executed'):
        raise ValueError('Official compiler, controls, inference and agent tests must pass the pipeline checks')
    if result['task_id'] != manifest['task_id']:
        raise ValueError('Smoke results differ from the prepared task')
    bundle = run / 'inputs/submission.zip'
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'competition': COMPETITION,
            'run_dir': str(run), 'archive': str(bundle), 'sha256': manifest['agent_archive_sha256'],
            'bytes': bundle.stat().st_size, 'smoke': result,
            'scope': 'One public-task smoke on the recorded platform. Full hidden scoring is determined by Kaggle.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--message', default='No-LoRA baseline; official compiler and public-task smoke checked')
    parser.add_argument('--submit', action='store_true', help='Actually consume a competition submission')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    result = check_run(run)
    if not args.submit:
        print(json.dumps(result, indent=2))
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    response = api.competitions_list(search=COMPETITION)
    competition = next(c for c in response.competitions if c.ref.rstrip('/').endswith('/' + COMPETITION))
    if competition.is_kernels_submissions_only or competition.submissions_disabled or not competition.user_has_entered:
        raise RuntimeError('Competition no longer permits this enrolled-account file-upload route')
    limits = api.competition_get_submission_limits(COMPETITION)
    if limits.num_allowed_now < 1:
        raise RuntimeError('No competition submissions remaining; no upload attempted')
    result.update(description=args.message, status='upload_started', submission_ref=None)
    receipt = run / 'competition-submission.json'
    # Never duplicate an upload after a timeout or uncertain response. Inspect
    # the existing receipt and Kaggle submission history before any retry.
    with receipt.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    try:
        response = api.competition_submit(str(result['archive']), args.message, COMPETITION, quiet=True)
        result.update(status='accepted' if response.ref else 'response_requires_inspection',
                      submission_ref=response.ref, response_message=response.message,
                      response_at=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        if not response.ref:
            raise RuntimeError('No submission reference returned; inspect history, do not blindly resubmit')
    except Exception as error:
        result.update(status='inspect_before_retry', error_type=type(error).__name__)
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
