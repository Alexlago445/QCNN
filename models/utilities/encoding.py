from __future__ import annotations

import math

import pennylane as qml
import torch


# -----------------------------------------------------------------------------
# Shared encoding metadata
# -----------------------------------------------------------------------------
E1_INIT_A = 0.2
LAMBDA_FUSION = math.pi / 4
E1_OMEGA_FIXED = math.pi / 2
BETA_GLOBAL = (1.0, 10.0, 10.0, 1.0)
E1_PARAMS = 16

N_PATCHES = 4
N_QUBITS_PER_PATCH = 2
N_REUPLOAD_LAYERS = 2
N_U_PER_REUPLOAD = 5
PARAMS_PER_U = 6  # 3 theta + 3 multiplicative data weights
REUPLOAD_PARAMS = (
    N_PATCHES
    * N_QUBITS_PER_PATCH
    * N_REUPLOAD_LAYERS
    * N_U_PER_REUPLOAD
    * PARAMS_PER_U
)  # 480

PATCH_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# Input size, qubit count and trainable parameter count for each encoding.
ENCODING_IMAGE_SIZE = {
    "reupload": 8,
    "e1": 16,
    "e3": 16,
}

ENCODING_QUBITS = {
    "reupload": 8,
    "e1": 9,  # 8 data wires + global ancilla on wire 8
    "e3": 8,
}

ENCODING_TRAINABLE_PARAMS = {
    "reupload": REUPLOAD_PARAMS,
    "e1": E1_PARAMS,
    "e3": 0,
}


# -----------------------------------------------------------------------------
# Spatial aware re-uploading encoder (8x8 input)
# -----------------------------------------------------------------------------

# Prepare image patches for the re-uploading encoder used by ConfigurableQCNN.
def patches_4x15(x_flat: torch.Tensor) -> torch.Tensor:
    """Return [B, 4, 15] features from four 4x4 quadrants of each 8x8 image.

    Drop the outer image corner of each quadrant, then scale [0,1] to [-1,1].
    """
    bsz = x_flat.shape[0]
    img = x_flat.reshape(bsz, 8, 8)

    quadrants = [
        img[:, 0:4, 0:4],
        img[:, 0:4, 4:8],
        img[:, 4:8, 0:4],
        img[:, 4:8, 4:8],
    ]
    drop_flat_indices = [0, 3, 12, 15]

    vectors = []
    for patch, drop_idx in zip(quadrants, drop_flat_indices):
        flat = patch.reshape(bsz, 16)
        keep = [i for i in range(16) if i != drop_idx]
        vec = flat[:, keep]
        vec = 2.0 * vec - 1.0  # [0,1] -> [-1,1]
        vectors.append(vec)

    return torch.stack(vectors, dim=1)

# Utility function to apply a basic SU(2) unitary used in the re-upload layers of the spatial aware re-uploading encoder.
def paper_u(phi: torch.Tensor, wire: int) -> None:
    """Basic SU(2) unitary used to construct the re-upload layers of the spatial aware re-uploading encoder."""
    qml.U3(phi[:, 0], -phi[:, 2], -phi[:, 1], wires=wire)

# Main function for the spatial aware re-uploading encoder, which applies two re-upload layers with a CZ gate between them for each spatial patch pair.
def data_reuploading_encoding(
    patches: torch.Tensor,
    theta_reupload: torch.Tensor,
    weight_reupload: torch.Tensor,
) -> None:
    """Two re-upload layers; one CZ between them for each spatial patch pair."""
    for patch_idx, (a, b) in enumerate(PATCH_PAIRS):
        # Split each 15-pixel patch into five groups of three features per image.
        chunks = patches[:, patch_idx, :].reshape(-1, 5, 3)

        for layer in range(N_REUPLOAD_LAYERS):  # Two re-upload layers per patch.
            for k in range(N_U_PER_REUPLOAD):  # Five feature-dependent U3 gates per qubit.
                xk = chunks[:, k, :]
                phi_a = (
                    theta_reupload[patch_idx, 0, layer, k]
                    + weight_reupload[patch_idx, 0, layer, k] * xk
                )
                phi_b = (
                    theta_reupload[patch_idx, 1, layer, k]
                    + weight_reupload[patch_idx, 1, layer, k] * xk
                )
                paper_u(phi_a, a)
                paper_u(phi_b, b)
            # Entangle the two qubits of this patch between re-upload layers.
            if layer < N_REUPLOAD_LAYERS - 1:
                qml.CZ(wires=[a, b])


# -----------------------------------------------------------------------------
# Amplitude encoding (16x16 input)
# -----------------------------------------------------------------------------
def amplitude_encoding_e3(x_flat: torch.Tensor) -> None:
    """Normalize each 256-pixel vector as amplitudes of the eight data qubits."""
    if x_flat.shape[-1] != 256:
        raise ValueError(
            f"E3 expects flattened 16x16 images (256 values), got {x_flat.shape[-1]}."
        )
    qml.AmplitudeEmbedding(features=x_flat, wires=range(8), normalize=True)


# -----------------------------------------------------------------------------
# E1 angle encoding with local means and global statistics (16x16 input)
# -----------------------------------------------------------------------------
def e1_quad_means_8(x_flat: torch.Tensor) -> torch.Tensor:
    """Return eight local means over the 2x4 grid of 8x4 pixel blocks."""
    if x_flat.shape[-1] != 256:
        raise ValueError(
            f"Expects flattened 16x16 images (256 values), got {x_flat.shape[-1]}."
        )
    img = x_flat.reshape(-1, 16, 16)
    blocks = [
        img[:, r * 8 : (r + 1) * 8, c * 4 : (c + 1) * 4].mean(dim=(1, 2))
        for r in range(2)
        for c in range(4)
    ]
    return torch.stack(blocks, dim=1)


def e1_global_stats_4(x_flat: torch.Tensor) -> torch.Tensor:
    """Return centered mean, variance, gradient energy and variance anisotropy."""
    if x_flat.shape[-1] != 256:
        raise ValueError(
            f"Expects flattened 16x16 images (256 values), got {x_flat.shape[-1]}."
        )
    img = x_flat.reshape(-1, 16, 16)
    gx = img[:, :, 1:] - img[:, :, :-1]
    gy = img[:, 1:, :] - img[:, :-1, :]
    hv = img.var(dim=2, unbiased=False).mean(dim=1)
    vv = img.var(dim=1, unbiased=False).mean(dim=1)
    return torch.stack(
        [
            img.mean(dim=(1, 2)) - 0.5,
            img.var(dim=(1, 2), unbiased=False),
            (gx.pow(2).mean(dim=(1, 2)) + gy.pow(2).mean(dim=(1, 2))) / 2,
            (hv - vv) / (hv + vv + 1e-8),
        ],
        dim=1,
    )


def compute_e1_gamma(global_stats_4: torch.Tensor) -> torch.Tensor:
    """Map the four global statistics to bounded angles for the E1 ancilla."""
    beta = global_stats_4.new_tensor(list(BETA_GLOBAL))
    return math.pi * torch.tanh(beta * global_stats_4)


def angle_encoding_e1_from_features(
    quad_means: torch.Tensor,
    gamma: torch.Tensor,
    a_embed: torch.Tensor,
    c_embed: torch.Tensor,
    ancilla: int = 8,
) -> None:
    """Apply E1 after its 16x16 classical preprocessing has been computed."""
    for i in range(8):
        qml.RY(a_embed[i] * (math.pi * quad_means[:, i]) + c_embed[i], wires=i)

    for fn, col in [(qml.RY, 0), (qml.RZ, 1), (qml.RX, 2), (qml.RZ, 3)]:
        fn(gamma[:, col], wires=ancilla)

    qml.RY(E1_OMEGA_FIXED, wires=ancilla)

    for fn, col in [(qml.RZ, 0), (qml.RX, 1), (qml.RY, 2), (qml.RZ, 3)]:
        fn(gamma[:, col], wires=ancilla)

    for i in range(8):
        qml.CNOT(wires=[ancilla, i])
        qml.RZ(LAMBDA_FUSION, wires=i)
        qml.CNOT(wires=[ancilla, i])

