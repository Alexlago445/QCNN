from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model19(ConfigurableQCNN):
    """benchmark topology + HUR8 + e3 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model19",
            topology="benchmark",
            ansatz_name="hur8",
            encoding_name="e3",
            init_scale=init_scale,
        )
