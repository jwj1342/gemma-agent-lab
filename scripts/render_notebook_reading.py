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
    scope = (f'另有 {pending} 项仍为结构扫描，继续补读。' if pending else
             '本次快照已全部逐项记录；其中空源码条目仅记录不可审阅的边界。')
    lines = ['# 公开 Notebook 逐项阅读表', '',
             f'当前 {len(index)} 个公开条目；已完成 {len(reviewed)} 项阅读记录。{scope}',
             '本表区分实际写入的提交内容、notebook 外围实验和作者自述。没有执行第三方 notebook，也不是逐行安全审计。',
             '原始下载不改写；嵌入的文字配置可以静态解码，不能因此认为其中脚本已运行或模型已训练。', '',
             '所有条目的原始入口见 [完整索引](README.md)，已有重点结论见 [比赛调研](../../docs/competition-research.md)。', '']
    for number, item in enumerate(index, 1):
        ref = item['ref']
        note = reviewed.get(ref)
        title = item['title'].replace('\n', ' ')
        lines += [f'## {number}. {title}', '',
                  f'[Kaggle](https://www.kaggle.com/code/{ref}) · [本地源码](public/{ref}/readable.txt)', '']
        if note is None:
            lines += ['状态：已下载和结构扫描；关键执行路径评语待补充。', '']
            continue
        source = BASE / 'public' / ref / 'readable.txt'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == note['readable_sha256'], ref
        count = len(source.read_text().splitlines())
        assert all(1 <= a <= b <= count for a, b in note['evidence_lines']), ref
        lines += [f"**类型：{note['role']}。** {note['implementation']}", '',
                  f"可借鉴：{note['reuse']}", '', f"边界与疑点：{note['caveats']}", '',
                  '阅读定位：' + (', '.join(f'L{a}–L{b}' for a, b in note['evidence_lines']) or '下载源码为空') + '。', '']
    (BASE / 'reading-notes.md').write_text('\n'.join(lines))
    print(json.dumps({'rendered_at': datetime.now(timezone.utc).isoformat(),
                      'reviewed_entries': len(reviewed), 'total_entries': len(index)}))


if __name__ == '__main__':
    main()
