from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model14(ConfigurableQCNN):
    """benchmark topology + HUR6 + reupload encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model14",
            topology="benchmark",
            ansatz_name="hur6",
            encoding_name="reupload",
            init_scale=init_scale,
        )
