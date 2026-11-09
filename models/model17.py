from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model17(ConfigurableQCNN):
    """benchmark topology + HUR8 + reupload encoding."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model17",
            topology="benchmark",
            ansatz_name="hur8",
            encoding_name="reupload",
            init_scale=init_scale,
        )
