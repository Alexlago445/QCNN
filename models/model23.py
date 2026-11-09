from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model23(ConfigurableQCNN):
    """spatial_diag topology + custom ansatz + reupload encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model23",
            topology="spatial_diag",
            ansatz_name="custom",
            encoding_name="reupload",
            init_scale=init_scale,
        )
