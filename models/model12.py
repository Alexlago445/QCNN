from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model12(ConfigurableQCNN):
    """spatial_diag topology + HUR9 + e1 encoding; HUR9 trace-out pooling."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model12",
            topology="spatial_diag",
            ansatz_name="hur9",
            encoding_name="e1",
            init_scale=init_scale,
        )
