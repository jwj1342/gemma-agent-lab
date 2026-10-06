"""Record full accelerator durations without the SDK to_dict() day-truncation bug."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('resource/competition/accelerator-quota-normalized.json'))
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    quota = api.quota_view()
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'refresh_at': quota.quota_refresh_time.isoformat(),
              'kagglesdk_version': version('kagglesdk'),
              'duration_conversion': 'timedelta.total_seconds(), never timedelta.seconds or SDK to_dict()'}
    for name, resource in [('gpu', quota.gpu_quota), ('tpu', quota.tpu_quota)]:
        if resource is None:
            continue
        fields = {field: getattr(resource, field).total_seconds() for field in
                  ['time_used', 'time_reserved', 'total_time_allowed', 'minimum_time_allowed']}
        fields['remaining_after_reservations'] = max(0, fields['total_time_allowed'] - fields['time_used'] - fields['time_reserved'])
        result[name + '_seconds'] = fields
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
