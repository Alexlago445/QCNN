from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model6(ConfigurableQCNN):
    """spatial_diag topology + HUR6 + e1 encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model6",
            topology="spatial_diag",
            ansatz_name="hur6",
            encoding_name="e1",
            init_scale=init_scale,
        )
