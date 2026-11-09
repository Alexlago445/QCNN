from __future__ import annotations

from .model5 import Model5
from .model6 import Model6
from .model7 import Model7
from .model8 import Model8
from .model9 import Model9
from .model10 import Model10
from .model11 import Model11
from .model12 import Model12
from .model13 import Model13
from .model14 import Model14
from .model15 import Model15
from .model16 import Model16
from .model17 import Model17
from .model18 import Model18
from .model19 import Model19
from .model20 import Model20
from .model21 import Model21
from .model22 import Model22
from .model23 import Model23
from .model24 import Model24
from .model25 import Model25
from .model26 import Model26
from .model27 import Model27
from .model28 import Model28


_MODELS = {
    "model5": Model5,
    "model6": Model6,
    "model7": Model7,
    "model8": Model8,
    "model9": Model9,
    "model10": Model10,
    "model11": Model11,
    "model12": Model12,
    "model13": Model13,
    "model14": Model14,
    "model15": Model15,
    "model16": Model16,
    "model17": Model17,
    "model18": Model18,
    "model19": Model19,
    "model20": Model20,
    "model21": Model21,
    "model22": Model22,
    "model23": Model23,
    "model24": Model24,
    "model25": Model25,
    "model26": Model26,
    "model27": Model27,
    "model28": Model28,
}


def get_model(name: str):
    try:
        model_cls = _MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS, key=lambda x: int(x[5:])))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc
    return model_cls()


def list_models() -> list[str]:
    return sorted(_MODELS, key=lambda x: int(x[5:]))
