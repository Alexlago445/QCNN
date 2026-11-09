from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model5(ConfigurableQCNN):
    """spatial_diag topology + HUR6 + reupload encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model5",
            topology="spatial_diag",
            ansatz_name="hur6",
            encoding_name="reupload",
            init_scale=init_scale,
        )
