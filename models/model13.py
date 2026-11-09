from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model13(ConfigurableQCNN):
    """spatial_diag topology + HUR9 + e3 encoding; HUR9 trace-out pooling."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model13",
            topology="spatial_diag",
            ansatz_name="hur9",
            encoding_name="e3",
            init_scale=init_scale,
        )
