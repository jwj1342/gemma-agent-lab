"""Render manually reviewed notebook notes; never infer review completion from keywords."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'resource/notebooks'


def main():
    index = json.loads((BASE / 'index.json').read_text())
    notes = json.loads((BASE / 'reading-notes.json').read_text())
    reviewed = {x['ref']: x for x in notes['entries']}
    assert len(reviewed) == len(notes['entries'])
    assert set(reviewed) <= {x['ref'] for x in index}
    pending = len(index) - len(reviewed)
    scope = (f'{pending} entries still have structure scans only and await detailed notes.' if pending else
             'Every entry has a reading record; the empty-source entry documents its review limitation.')
    lines = ['# Public notebook reading notes', '',
             f'Snapshot: {len(index)} public entries, {len(reviewed)} individual reading records. {scope}',
             'These notes distinguish submitted payloads, surrounding notebook experiments, and author claims. No third-party notebook was executed; this is not a line-by-line security audit.',
             'Original downloads remain unchanged. Static decoding of embedded configuration does not establish that its scripts ran or its model was trained.', '',
             'See the [complete index](README.md) for sources and [competition research](../../docs/competition-research.md) for conclusions.', '']
    for number, item in enumerate(index, 1):
        ref = item['ref']
        note = reviewed.get(ref)
        title = item['title'].replace('\n', ' ').strip()
        lines += [f'## {number}. {title}', '',
                  f'[Kaggle](https://www.kaggle.com/code/{ref}) · [Local source](public/{ref}/readable.txt)', '']
        if note is None:
            lines += ['Status: downloaded and structurally scanned; execution-path notes are pending.', '']
            continue
        source = BASE / 'public' / ref / 'readable.txt'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == note['readable_sha256'], ref
        count = len(source.read_text().splitlines())
        assert all(1 <= a <= b <= count for a, b in note['evidence_lines']), ref
        lines += [f"**Category: {note['role']}.** {note['implementation']}", '',
                  f"Reusable ideas: {note['reuse']}", '', f"Caveats: {note['caveats']}", '',
                  'Source locations: ' + (', '.join(f'L{a}–L{b}' for a, b in note['evidence_lines']) or 'Downloaded source is empty') + '.', '']
    (BASE / 'reading-notes.md').write_text('\n'.join(lines))
    print(json.dumps({'rendered_at': datetime.now(timezone.utc).isoformat(),
                      'reviewed_entries': len(reviewed), 'total_entries': len(index)}))


if __name__ == '__main__':
    main()
