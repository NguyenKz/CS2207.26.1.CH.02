"""Single source of truth for the four MNIST demo model configurations.

The notebooks copy ``MODEL_CONFIGS`` before tuning so that a tuning run can
update the size-specific JSON config without changing these defaults.
"""

from __future__ import annotations

import numpy as np

from .digits_config import EARLY_STOPPING, MAX_ITER, PIXEL_COUNT, PIXEL_SIZE


RANDOM_SEED = 42
CLASS_COUNT = 10
SEARCH_ROUNDS = 30
SEARCH_MAX_ITER = 120
TUNING_SEARCH_SAMPLE_LIMIT = 24000
AUGMENT_TRAINING = True
AUGMENT_FACTOR = 3
AUGMENT_SHIFT_PIXELS = 2
AUGMENT_STROKE_VARIANTS = True
AUGMENT_SCALE_RANGE = (0.90, 1.10)
BATCH_SIZE = 256
LOGISTIC_MAX_ITER = 300
LOGISTIC_CANDIDATES = tuple(float(value) for value in np.logspace(-4, 2, SEARCH_ROUNDS))
MLP_SEARCH_CONFIGS = tuple(
    (architecture, "relu", learning_rate)
    for architecture in ((16,), (24,), (32,), (40,), (48,), (32, 16), (48, 24))
    for learning_rate in (0.0005, 0.0007, 0.001, 0.002)
) + (((48, 24), "tanh", 0.001), ((32,), "tanh", 0.001))


# Edit this dictionary when changing the demo architecture. The three
# notebooks and the backend training helper import the same defaults.
MODEL_CONFIGS = {
    "logistic": {
        "kind": "linear",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": LOGISTIC_MAX_ITER,
    },
    "mlp_4_one_layer": {
        "kind": "ann",
        "hidden_layer_sizes": (PIXEL_SIZE // 2,),
        "activation": "tanh",
        "learning_rate_init": 0.001,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
    },
    "mlp_4_two_layers": {
        "kind": "ann",
        "hidden_layer_sizes": (PIXEL_SIZE // 2, PIXEL_SIZE // 2),
        "activation": "tanh",
        "learning_rate_init": 0.001,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
    },
    "compact_tuned": {
        "kind": "ann",
        "hidden_layer_sizes": (PIXEL_COUNT // 2, PIXEL_COUNT // 4),
        "activation": "relu",
        "learning_rate_init": 0.002,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
    },
}
