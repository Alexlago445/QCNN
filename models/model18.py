from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model18(ConfigurableQCNN):
    """benchmark topology + HUR8 + e1 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model18",
            topology="benchmark",
            ansatz_name="hur8",
            encoding_name="e1",
            init_scale=init_scale,
        )
