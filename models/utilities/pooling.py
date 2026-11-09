from __future__ import annotations

import pennylane as qml
import torch


# -----------------------------------------------------------------------------
# Number of trainable parameters in a single pooling block.
HUR_POOL_PARAMS = 2
CUSTOM_POOL_PARAMS = 3

# -----------------------------------------------------------------------------
# Pooling circuit definitions, used to couple discarded and surviving qubits
# before the next QCNN stage. Each pair is ordered as (discard, keep).

def hur_pool(theta_pool: torch.Tensor, ctrl: int, tgt: int) -> None:
    """HUR6/HUR8 pooling: two parameters, with ctrl discarded and tgt kept."""
    qml.CRZ(theta_pool[0], wires=[ctrl, tgt])
    qml.PauliX(wires=ctrl)
    qml.CRX(theta_pool[1], wires=[ctrl, tgt])
    qml.PauliX(wires=ctrl)


def custom_pool(theta_pool: torch.Tensor, discard: int, keep: int) -> None:
    """Custom pooling: controlled interactions and a local rotation, with 3 parameters."""
    qml.CNOT(wires=[discard, keep])
    qml.CRY(theta_pool[0], wires=[discard, keep])
    qml.CRZ(theta_pool[1], wires=[discard, keep])
    qml.RY(theta_pool[2], wires=keep)


def no_trainable(theta_pool: torch.Tensor, discard: int, keep: int) -> None:
    """HUR9 trace-out pooling: add no gates; later stages use only surviving wires."""
    return None


# -----------------------------------------------------------------------------
# Utility functions for pooling selection and application, used in
# models/utilities/qcnn_model.py to construct the two QCNN reduction stages.

# Apply the pooling block across qubit pairs with shared parameters.
def apply_pool(pool_fn, theta_pool: torch.Tensor, pairs) -> None:
    """Apply pooling to each (discard, keep) pair in the supplied order."""
    for discard, keep in pairs:
        pool_fn(theta_pool, discard, keep)


# Select the pooling function and its resource counts from the ansatz name.
def get_pooling(ansatz_name: str):
    """Return (function, n_params, logical_2q_per_application)."""
    if ansatz_name in {"hur6", "hur8"}:
        pool_fn, n_params = hur_pool, HUR_POOL_PARAMS
    elif ansatz_name == "hur9":
        pool_fn, n_params = no_trainable, 0
    elif ansatz_name == "custom":
        pool_fn, n_params = custom_pool, CUSTOM_POOL_PARAMS
    else:
        raise ValueError(f"No pooling rule defined for ansatz '{ansatz_name}'.")
    return pool_fn, n_params, POOL_2Q_PER_APPLICATION[pool_fn.__name__]


# -----------------------------------------------------------------------------
# Logical two-qubit operations per pooling block, before hardware compilation.
POOL_2Q_PER_APPLICATION = {
    "hur_pool": 2,     # CRZ + CRX
    "no_trainable": 0, # no pooling gates; discarded wires are simply not reused
    "custom_pool": 3,  # CNOT + CRY + CRZ
}
