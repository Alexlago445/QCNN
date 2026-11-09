from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model11(ConfigurableQCNN):
    """spatial_diag topology + HUR9 + reupload encoding; HUR9 trace-out pooling."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model11",
            topology="spatial_diag",
            ansatz_name="hur9",
            encoding_name="reupload",
            init_scale=init_scale,
        )
