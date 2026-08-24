"""Trusted plugin example: favor points near the current pool center."""

import numpy as np

from agentic_al import register_acquisition


def center_seeking(context, config):
    del config
    center = context.pool_features.mean(axis=0)
    distance = np.linalg.norm(context.pool_features - center, axis=1)
    return 1.0 / (1.0 + distance)


register_acquisition("center_seeking", center_seeking)
