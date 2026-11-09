from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model27(ConfigurableQCNN):
    """benchmark topology + custom ansatz + e1 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model27",
            topology="benchmark",
            ansatz_name="custom",
            encoding_name="e1",
            init_scale=init_scale,
        )
