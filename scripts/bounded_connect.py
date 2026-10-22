"""Run a command with a bounded per-address TCP connect timeout.

Why this exists
---------------
This machine has a global IPv6 address on the campus WLAN, and reachability to Google is
per-address rather than per-family: `curl` reached `storage.googleapis.com` over IPv6 in
1.0 s while a Python upload sat in `SYN_SENT` to `172.217.117.4`. Python's HTTP stacks walk
the resolver's address list one address at a time, and with no explicit timeout each
attempt waits for the operating-system default (about 21 s on Windows). A host that
resolves to a dozen addresses can therefore spend minutes before reaching a working one,
which is what the recorded Kaggle dataset-upload stalls looked like (40 minutes via the
CLI, 20 minutes via the SDK at ~0.9 s CPU, no files uploaded).

`urllib3.util.connection.create_connection` already loops over every resolved address, so
the only missing piece is a short cap on each attempt. Patching `socket.socket.connect`
supplies that for every stack that uses blocking connects (requests/urllib3, httpcore, and
plain sockets), and the socket is returned to blocking mode once connected so that data
transfer keeps the caller's timeout-free behaviour.

The patch is process-local; it changes nothing on the host and needs no administrator
rights.

Usage
-----
    python scripts/bounded_connect.py -m kaggle datasets create -p <dir> --dir-mode zip
    python scripts/bounded_connect.py --log -m kaggle datasets list --mine
    python scripts/bounded_connect.py scripts/check_kaggle_quota.py --output <path>
"""
import runpy
import socket
import sys
import time

CONNECT_TIMEOUT = 5.0

_real_connect = socket.socket.connect
_log = False


def _describe(address):
    if isinstance(address, tuple) and len(address) >= 2:
        return f'{address[0]}:{address[1]}'
    return str(address)


def connect_with_timeout(self, address):
    """Bound a blocking connect to CONNECT_TIMEOUT, then restore blocking mode."""
    if self.gettimeout() is not None:
        return _real_connect(self, address)
    started = time.time()
    self.settimeout(CONNECT_TIMEOUT)
    try:
        _real_connect(self, address)
    except OSError as exc:
        if _log:
            print(f'connect {_describe(address)} failed after {time.time() - started:.1f}s: '
                  f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
    finally:
        self.settimeout(None)
    if _log:
        print(f'connect {_describe(address)} ok in {time.time() - started:.1f}s',
              file=sys.stderr, flush=True)


def main(argv):
    global _log
    if '--log' in argv:
        argv = [item for item in argv if item != '--log']
        _log = True
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    socket.socket.connect = connect_with_timeout
    if argv[0] == '-m':
        if len(argv) < 2:
            print('bounded_connect.py: -m requires a module name', file=sys.stderr)
            return 2
        sys.argv = argv[1:]
        if _log:
            print(f'bounded_connect: cap {CONNECT_TIMEOUT}s per address, running {argv[1]}',
                  file=sys.stderr, flush=True)
        runpy.run_module(argv[1], run_name='__main__', alter_sys=True)
    else:
        sys.argv = argv
        runpy.run_path(argv[0], run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
