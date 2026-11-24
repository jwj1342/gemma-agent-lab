"""Prepare sample controls using the same strict JUnit gate as the full sweep.

The existing --tasks, --ids, --out, --owner, --slug and --title options are supported.
The shared generator writes sweep.ipynb and CPU-only kernel metadata.
"""
from make_mined_sweep_notebook import main


if __name__ == '__main__':
    raise SystemExit(main())
