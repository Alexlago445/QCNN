from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model15(ConfigurableQCNN):
    """benchmark topology + HUR6 + e1 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model15",
            topology="benchmark",
            ansatz_name="hur6",
            encoding_name="e1",
            init_scale=init_scale,
        )
