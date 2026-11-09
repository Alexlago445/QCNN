from __future__ import annotations

import pennylane as qml
import torch


# -----------------------------------------------------------------------------
# Number of trainable parameters in a single two-qubit building block.
HUR6_PARAMS = 6
HUR8_PARAMS = 10
HUR9_PARAMS = 15
CUSTOM_PARAMS = 11

ANSATZ_PARAMS = {
    "hur6": HUR6_PARAMS,
    "hur8": HUR8_PARAMS,
    "hur9": HUR9_PARAMS,
    "custom": CUSTOM_PARAMS,
}

# -----------------------------------------------------------------------------
# Two-qubit building blocks circuit definitions.
# Used to construct the convolutional ansatz of the QCNN models. 
# Theta holds the angles, wires identifies the qubit pair.
def hur6(theta: torch.Tensor, wires) -> None:
    """HUR6 SO(4) block: RY rotations and two CNOTs, with 6 parameters."""
    a, b = wires
    qml.RY(theta[0], wires=a)
    qml.RY(theta[1], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[2], wires=a)
    qml.RY(theta[3], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[4], wires=a)
    qml.RY(theta[5], wires=b)


def hur8(theta: torch.Tensor, wires) -> None:
    """qcnn_benchmark HUR8 block: RX/RZ rotations and one CNOT, with 10 parameters."""
    a, b = wires
    qml.RX(theta[0], wires=a)
    qml.RX(theta[1], wires=b)
    qml.RZ(theta[2], wires=a)
    qml.RZ(theta[3], wires=b)
    qml.RX(theta[4], wires=a)
    qml.RX(theta[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RX(theta[6], wires=a)
    qml.RX(theta[7], wires=b)
    qml.RZ(theta[8], wires=a)
    qml.RZ(theta[9], wires=b)


def hur9(theta: torch.Tensor, wires) -> None:
    """qcnn_benchmark HUR9 SU(4)/KAK block: 15 parameters and three CNOTs."""
    a, b = wires
    qml.U3(theta[0], theta[1], theta[2], wires=a)
    qml.U3(theta[3], theta[4], theta[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[6], wires=a)
    qml.RZ(theta[7], wires=b)
    qml.CNOT(wires=[b, a])
    qml.RY(theta[8], wires=a)
    qml.CNOT(wires=[a, b])
    qml.U3(theta[9], theta[10], theta[11], wires=a)
    qml.U3(theta[12], theta[13], theta[14], wires=b)


def custom_ansatz(theta: torch.Tensor, wires) -> None:
    """qcnn_benchmark custom block: local rotations and XX/YY/ZZ interactions, with 11 parameters."""
    a, b = wires
    qml.RZ(theta[0], wires=a)
    qml.RY(theta[1], wires=a)
    qml.RZ(theta[2], wires=b)
    qml.RY(theta[3], wires=b)
    qml.IsingXX(theta[4], wires=[a, b])
    qml.IsingYY(theta[5], wires=[a, b])
    qml.IsingZZ(theta[6], wires=[a, b])
    qml.RY(theta[7], wires=a)
    qml.RZ(theta[8], wires=a)
    qml.RY(theta[9], wires=b)
    qml.RZ(theta[10], wires=b)


# -----------------------------------------------------------------------------
# Utility functions for ansatz selection and application, used in models/utilities/qcnn_model.py to 
# construct the convolutional ansatz of the QCNN models. 

# Map each ansatz name to its block function.
ANSATZ_FUNCTIONS = {
    "hur6": hur6,
    "hur8": hur8,
    "hur9": hur9,
    "custom": custom_ansatz,
}

# Select the ansatz function and its parameter count by name.
def get_ansatz(name: str):
    """Return the block function and parameter count; reject unknown names."""
    try:
        return ANSATZ_FUNCTIONS[name], ANSATZ_PARAMS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ansatz '{name}'. Available: {sorted(ANSATZ_FUNCTIONS)}"
        ) from exc


# Apply the ansatz across qubit pairs with shared parameters.
def apply_shared(ansatz_fn, theta: torch.Tensor, pairs) -> None:
    """Apply the block to each pair in order, sharing the same theta vector."""
    for pair in pairs:
        ansatz_fn(theta, pair)

# -----------------------------------------------------------------------------
# Logical two-qubit operations per block, before hardware compilation.
ANSATZ_2Q_PER_APPLICATION = {
    "hur6": 2,   # 2 CNOT
    "hur8": 1,   # 1 CNOT
    "hur9": 3,   # 3 CNOT
    "custom": 3, # IsingXX + IsingYY + IsingZZ
}

