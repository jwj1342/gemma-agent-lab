"""Build a ``CompressedTensorsConfig`` with the keyword the *installed* transformers understands.

``transformers`` renamed this option between the version the wheelhouse ships and the version the H100
recipe ran:

* the wheelhouse's ``5.13.1`` takes ``run_compressed: bool = True`` and reports
  ``get_loading_attributes() == {'run_compressed': ...}``;
* ``5.18``/``5.19`` take ``dequantize: bool = False`` and describe ``run_compressed`` as deprecated with
  "``run_compressed=False`` meaning ``dequantize=True``".

Passing the newer keyword to the older class does not fail. ``5.13.1`` accepts ``**kwargs``, uses it only
when ``config_groups`` or ``kv_cache_scheme`` is given (neither is, here), and therefore **drops it**: the
checkpoint stays compressed and the caller believes the base was dequantised to BF16. A whole training
run could then measure the wrong base while recording the right intention, which is exactly the kind of
silent error this project's rules exist to prevent. This module makes the request explicit and verifies
the effective value, refusing to return a config whose loading mode disagrees with the request.
"""
from __future__ import annotations

import inspect


def build_quantization_config(config_class, dequantize: bool):
    """Return ``(config, record)`` for the requested loading mode.

    ``dequantize=True`` asks for the BF16 dense base the recipe trains; ``False`` asks for the checkpoint
    as shipped. Raises ``TypeError`` when the class exposes neither spelling, and ``ValueError`` when the
    constructed config reports a different mode than the one requested.
    """
    parameters = inspect.signature(config_class.__init__).parameters
    record = {'requested_dequantize': dequantize}
    if 'dequantize' in parameters:
        config = config_class(dequantize=dequantize)
        record['keyword'] = 'dequantize'
    elif 'run_compressed' in parameters:
        # The pre-rename spelling: `run_compressed=False` is `dequantize=True`.
        config = config_class(run_compressed=not dequantize)
        record['keyword'] = 'run_compressed'
    else:
        raise TypeError(f'{config_class.__name__} accepts neither `dequantize` nor `run_compressed`; '
                        f'its parameters are {sorted(parameters)}')
    if hasattr(config, 'dequantize'):
        record['effective_dequantize'] = bool(config.dequantize)
    elif hasattr(config, 'run_compressed'):
        record['effective_dequantize'] = not bool(config.run_compressed)
    else:
        record['effective_dequantize'] = None
    if record['effective_dequantize'] is not dequantize:
        raise ValueError(f'requested dequantize={dequantize} but {config_class.__name__} reports '
                         f'effective_dequantize={record["effective_dequantize"]}; refusing to load a '
                         'base whose loading mode is not the one asked for')
    return config, record
