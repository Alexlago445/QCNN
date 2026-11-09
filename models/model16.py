from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model16(ConfigurableQCNN):
    """benchmark topology + HUR6 + e3 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model16",
            topology="benchmark",
            ansatz_name="hur6",
            encoding_name="e3",
            init_scale=init_scale,
        )
