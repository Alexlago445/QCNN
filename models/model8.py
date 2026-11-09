from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model8(ConfigurableQCNN):
    """spatial_diag topology + HUR8 + reupload encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model8",
            topology="spatial_diag",
            ansatz_name="hur8",
            encoding_name="reupload",
            init_scale=init_scale,
        )
