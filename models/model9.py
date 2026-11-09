from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model9(ConfigurableQCNN):
    """spatial_diag topology + HUR8 + e1 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model9",
            topology="spatial_diag",
            ansatz_name="hur8",
            encoding_name="e1",
            init_scale=init_scale,
        )
