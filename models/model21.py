from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model21(ConfigurableQCNN):
    """benchmark topology + HUR9 + e1 encoding; HUR9 trace-out pooling."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model21",
            topology="benchmark",
            ansatz_name="hur9",
            encoding_name="e1",
            init_scale=init_scale,
        )
