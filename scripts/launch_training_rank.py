"""Restrict each torchrun worker to one allocated GPU before importing CUDA libraries."""
import os
import sys


def isolated_environment(environment):
    """Keep global ranks unchanged while making the assigned GPU local device zero."""
    visible = environment.get('CUDA_VISIBLE_DEVICES', '')
    devices = [value.strip() for value in visible.split(',')]
    if not visible or any(not value or value == '-1' for value in devices):
        raise ValueError('An explicit nonempty CUDA_VISIBLE_DEVICES allocation is required')
    if len(devices) != len(set(devices)):
        raise ValueError('Allocated GPU identifiers must be unique')
    try:
        local_rank = int(environment['LOCAL_RANK'])
    except (KeyError, ValueError) as error:
        raise ValueError('Run this launcher through torchrun with an integer LOCAL_RANK') from error
    if not 0 <= local_rank < len(devices):
        raise ValueError('Worker local rank is outside the visible GPU allocation')
    result = dict(environment)
    result['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
    result['GEMMA_WORKER_LOCAL_RANK'] = str(local_rank)
    result['LOCAL_RANK'] = '0'
    return result


def main():
    environment = isolated_environment(os.environ)
    print(f"Global rank {environment.get('RANK')}: allocated local rank "
          f"{environment['GEMMA_WORKER_LOCAL_RANK']}, GPU "
          f"{environment['CUDA_VISIBLE_DEVICES']}, process-local CUDA device 0", flush=True)
    # A fresh interpreter ensures no imported dependency has already initialized CUDA.
    os.execve(sys.executable, [sys.executable, '-m', 'scripts.train_swe_lora', *sys.argv[1:]], environment)


if __name__ == '__main__':
    main()
