from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model25(ConfigurableQCNN):
    """spatial_diag topology + custom ansatz + e3 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model25",
            topology="spatial_diag",
            ansatz_name="custom",
            encoding_name="e3",
            init_scale=init_scale,
        )
