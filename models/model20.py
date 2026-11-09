from __future__ import annotations

from .utilities.qcnn_model import ConfigurableQCNN


class Model20(ConfigurableQCNN):
    """benchmark topology + HUR9 + reupload encoding; HUR9 trace-out pooling."""

    def __init__(self, init_scale: float = 0.01):
        super().__init__(
            model_name="model20",
            topology="benchmark",
            ansatz_name="hur9",
            encoding_name="reupload",
            init_scale=init_scale,
        )
