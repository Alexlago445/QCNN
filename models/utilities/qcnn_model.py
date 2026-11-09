from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

# Import the ansatz, encoding, and pooling from the other utility modules. 
from .ansatz import ANSATZ_2Q_PER_APPLICATION, apply_shared, get_ansatz
from .encoding import (
    ENCODING_IMAGE_SIZE,
    ENCODING_QUBITS,
    ENCODING_TRAINABLE_PARAMS,
    E1_INIT_A,
    N_PATCHES,
    N_QUBITS_PER_PATCH,
    N_REUPLOAD_LAYERS,
    N_U_PER_REUPLOAD,
    amplitude_encoding_e3,
    angle_encoding_e1_from_features,
    compute_e1_gamma,
    data_reuploading_encoding,
    e1_global_stats_4,
    e1_quad_means_8,
    patches_4x15,
)
from .pooling import apply_pool, get_pooling



#-----------------------------------------------------------------------------
# Connectivity definitions for the two QCNN topologies, including convolution pairs, pooling pairs and output wires.

# Output wires and pooling pairs shared by all models.
# Pooling pairs are ordered as (discard, keep), reducing active data wires 8 -> 4 -> 2.
OUTPUT_WIRES = [0, 4]
POOL_STAGE1 = [(1, 0), (3, 2), (5, 4), (7, 6)]
POOL_STAGE2 = [(2, 0), (6, 4)]


# Spatial-aware topology -->  horizontal/vertical interactions at the fine stage,
# followed by horizontal/vertical/diagonal interactions on four surviving wires.
SPATIAL_FINE_HORIZONTAL = [(0, 2), (1, 3), (4, 6), (5, 7)]
SPATIAL_FINE_VERTICAL = [(0, 4), (1, 5), (2, 6), (3, 7)]
SPATIAL_COARSE_HORIZONTAL = [(0, 2), (4, 6)]
SPATIAL_COARSE_VERTICAL = [(0, 4), (2, 6)]
SPATIAL_COARSE_DIAGONAL = [(0, 6), (2, 4)]


# Connectivity helper used to build the classic convolution topology.
def chain_pairs(active_wires) -> list[tuple[int, int]]:
    """Classic convolution order: even pairs followed by shifted pairs."""
    wires = list(active_wires)
    n = len(wires)
    even = [(wires[i], wires[i + 1]) for i in range(0, n - 1, 2)]
    shifted = [(wires[i], wires[i + 1]) for i in range(1, n - 1, 2)]
    return even + shifted

# Classic topology -->  alternate pairs followed by shifted pairs at each stage.
BENCHMARK_W8 = [0, 1, 2, 3, 4, 5, 6, 7]
BENCHMARK_W4 = [0, 2, 4, 6]
BENCHMARK_FINE_PAIRS = chain_pairs(BENCHMARK_W8)      # 7 applications
BENCHMARK_COARSE_PAIRS = chain_pairs(BENCHMARK_W4)   # 3 applications


# -----------------------------------------------------------------------------
# Shared PyTorch model used by all models to combine topology,
# convolution blocks, pooling rules and image encoding in one QCNN.
# inherit from nn.Module to integrate with PyTorch training and evaluation.

class ConfigurableQCNN(nn.Module):
    """Shared implementation for the modular QCNN variants.

    A concrete model fixes:
      - topology: ``spatial_diag`` or ``benchmark``;
      - ansatz: ``hur6``, ``hur8``, ``hur9`` or ``custom``;
      - encoding: ``reupload``, ``e1`` or ``e3``.

    Pooling is selected by ansatz for both topologies:
      - HUR6/HUR8 -> two-parameter HUR pooling;
      - HUR9 -> trace-out pooling without gates or trainable parameters;
      - custom -> three-parameter transfer pooling.
    """

    n_classes = 4

    def __init__(
        self,
        *,
        model_name: str,
        topology: str,
        ansatz_name: str,
        encoding_name: str,
        init_scale: float = 0.01,
    ):
        super().__init__()

        # Validate the configuration and retrieve the convolution/pooling rules.
        if topology not in {"spatial_diag", "benchmark"}:
            raise ValueError(f"Unknown topology '{topology}'.")
        if encoding_name not in ENCODING_IMAGE_SIZE:
            raise ValueError(f"Unknown encoding '{encoding_name}'.")

        conv_fn, n_conv = get_ansatz(ansatz_name)
        pool_fn, n_pool, pool_2q = get_pooling(ansatz_name)

        # Store circuit components and metadata used by training and result files.
        self.model_name = model_name
        self.topology = topology
        self.ansatz_name = ansatz_name
        self.encoding_name = encoding_name
        self.conv_fn = conv_fn
        self.n_conv = n_conv
        self.pool_fn = pool_fn
        self.n_pool = n_pool
        self.pool_2q_per_application = pool_2q
        self.use_pool = n_pool > 0
        self.image_size = ENCODING_IMAGE_SIZE[encoding_name]
        self.n_qubits = ENCODING_QUBITS[encoding_name]

        # Initialize encoding parameters, then convolution and pooling parameters.
        self._init_encoding(init_scale)
        self._init_qkernel(init_scale)

        # Count of trainable parameters: Two convolution vectors, two pooling vectors and the encoding parameters.
        self.expected_n_params = (
            2 * self.n_conv
            + 2 * self.n_pool
            + ENCODING_TRAINABLE_PARAMS[self.encoding_name]
        )

        self.architecture_name = f"{self.topology}_{self.ansatz_name}_{self.encoding_name}"

        # Create the state simulator and the differentiable circuit callable.
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = self._build_qnode() # construct the QNode based on the selected encoding and circuit structure

    # -------------------------------------------------------------------------
    # Utility functions to initialize convolution/pooling parameters and encoding parameters, called by the constructor of ConfigurableQCNN.

    # Initialize convolution and pooling parameters near zero.
    def _init_qkernel(self, init_scale: float) -> None:
        self.theta_fine = nn.Parameter(init_scale * torch.randn(self.n_conv))
        self.theta_coarse = nn.Parameter(init_scale * torch.randn(self.n_conv))

        if self.use_pool:
            self.theta_pool1 = nn.Parameter(init_scale * torch.randn(self.n_pool))
            self.theta_pool2 = nn.Parameter(init_scale * torch.randn(self.n_pool))
        else:
            # Trace-out pooling has no trainable parameters; keep empty buffers.
            self.register_buffer("theta_pool1", torch.empty(0))
            self.register_buffer("theta_pool2", torch.empty(0))

    # Initialize only the parameters required by the selected image encoding.
    def _init_encoding(self, init_scale: float) -> None:
        if self.encoding_name == "reupload":
            # Axes: patch, qubit within patch, upload layer, U3 gate, angle component.
            shape = (
                N_PATCHES,
                N_QUBITS_PER_PATCH,
                N_REUPLOAD_LAYERS,
                N_U_PER_REUPLOAD,
                3,
            )
            self.theta_reupload = nn.Parameter(init_scale * torch.randn(*shape))
            self.weight_reupload = nn.Parameter(1.0 + init_scale * torch.randn(*shape))
            self.a_embed = None
            self.c_embed = None

        elif self.encoding_name == "e1":
            self.theta_reupload = None
            self.weight_reupload = None
            self.a_embed = nn.Parameter(torch.full((8,), E1_INIT_A))
            self.c_embed = nn.Parameter(torch.zeros(8))

        else:  # e3
            self.theta_reupload = None
            self.weight_reupload = None
            self.a_embed = None
            self.c_embed = None

    # -------------------------------------------------------------------------
    # Apply convolution blocks using the connectivity defined by self.topology.
    # The fine stage acts on all eight data qubits; the coarse stage acts on [0, 2, 4, 6].
    # Each stage shares its own parameter vector across all qubit pairs.
    # All utility functions are called by the QNode to build the circuit.

    # Apply the first convolution stage on all eight data wires.
    def _apply_fine(self, theta) -> None:
        if self.topology == "spatial_diag":
            apply_shared(self.conv_fn, theta, SPATIAL_FINE_HORIZONTAL)
            apply_shared(self.conv_fn, theta, SPATIAL_FINE_VERTICAL)
        else:
            apply_shared(self.conv_fn, theta, BENCHMARK_FINE_PAIRS)

    # Apply the second convolution stage on surviving wires [0, 2, 4, 6].
    def _apply_coarse(self, theta) -> None:
        if self.topology == "spatial_diag":
            apply_shared(self.conv_fn, theta, SPATIAL_COARSE_HORIZONTAL)
            apply_shared(self.conv_fn, theta, SPATIAL_COARSE_VERTICAL)
            apply_shared(self.conv_fn, theta, SPATIAL_COARSE_DIAGONAL)
        else:
            apply_shared(self.conv_fn, theta, BENCHMARK_COARSE_PAIRS)

    # Apply the first pooling stage, retaining four active data wires.
    def _apply_pool1(self, theta_pool) -> None:
        apply_pool(self.pool_fn, theta_pool, POOL_STAGE1)

    # Apply the second pooling stage, retaining output wires [0, 4].
    def _apply_pool2(self, theta_pool) -> None:
        apply_pool(self.pool_fn, theta_pool, POOL_STAGE2)

    # Build and return a QNode: a PennyLane callable that executes the quantum circuit
    # and supports gradient computation through PyTorch.
    # All variants follow : encoding -> fine conv -> pool -> coarse conv -> pool, then return four class probabilities from output wires [0, 4].

    def _build_qnode(self):
        if self.encoding_name == "reupload":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(
                pts,
                theta_reupload,
                weight_reupload,
                fine,
                p1,
                coarse,
                p2,
            ):
                data_reuploading_encoding(pts, theta_reupload, weight_reupload)
                self._apply_fine(fine)
                self._apply_pool1(p1)
                self._apply_coarse(coarse)
                self._apply_pool2(p2)
                return qml.probs(wires=OUTPUT_WIRES)

        elif self.encoding_name == "e1":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(
                qm8,
                gamma4,
                a_embed,
                c_embed,
                fine,
                p1,
                coarse,
                p2,
            ):
                angle_encoding_e1_from_features(qm8, gamma4, a_embed, c_embed)
                self._apply_fine(fine)
                self._apply_pool1(p1)
                self._apply_coarse(coarse)
                self._apply_pool2(p2)
                return qml.probs(wires=OUTPUT_WIRES)

        else:  # e3
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(x_flat, fine, p1, coarse, p2):
                amplitude_encoding_e3(x_flat)
                self._apply_fine(fine)
                self._apply_pool1(p1)
                self._apply_coarse(coarse)
                self._apply_pool2(p2)
                return qml.probs(wires=OUTPUT_WIRES)

        return circuit

    # -------------------------------------------------------------------------
    # Forward pass, invoked by PyTorch when training/engine.py calls model(x_flat).
    # Prepare encoding-specific features and execute the QNode with current parameters.
    # Input: flattened images [B, 64] for reupload or [B, 256] for E1/E3.
    # Output: float32 class probabilities [B, 4], used during training and evaluation.

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self.encoding_name == "reupload":
            # Reupload: form four 15-pixel patches per image before circuit execution.
            pts = patches_4x15(x_flat)
            return self.qnode(
                pts,
                self.theta_reupload,
                self.weight_reupload,
                self.theta_fine,
                self.theta_pool1,
                self.theta_coarse,
                self.theta_pool2,
            ).float()

        if self.encoding_name == "e1":
            # E1: compute eight local means and four angles from global image statistics.
            qm8 = e1_quad_means_8(x_flat)
            gamma4 = compute_e1_gamma(e1_global_stats_4(x_flat))
            return self.qnode(
                qm8,
                gamma4,
                self.a_embed,
                self.c_embed,
                self.theta_fine,
                self.theta_pool1,
                self.theta_coarse,
                self.theta_pool2,
            ).float()

        # E3: pass all 256 pixels directly to the circuit for amplitude embedding.
        return self.qnode(
            x_flat,
            self.theta_fine,
            self.theta_pool1,
            self.theta_coarse,
            self.theta_pool2,
        ).float()

    # -------------------------------------------------------------------------
    # Parameter utilities used by the training engine for counts and optimizer setup.

    # Count all registered tensor elements that require gradients.
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Separate convolution/pooling parameters from trainable encoding parameters.
    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        qkernel = [self.theta_fine, self.theta_coarse]
        if self.use_pool:
            qkernel.extend([self.theta_pool1, self.theta_pool2])

        if self.encoding_name == "reupload":
            embedding = [self.theta_reupload, self.weight_reupload]
        elif self.encoding_name == "e1":
            embedding = [self.a_embed, self.c_embed]
        else:
            embedding = []

        return {"qkernel": qkernel, "embedding": embedding}

    # -------------------------------------------------------------------------
    # Identify the model variant by encoding, topology and ansatz.
    # These names are saved in the experiment configuration for resume checks.

    def model_config(self) -> dict:
        return {
            "encoding": self.encoding_name,
            "topology": self.topology,
            "ansatz": self.ansatz_name,
        }

    # -------------------------------------------------------------------------
    # Logical resource accounting for model comparison, before hardware compilation.
    # Count block applications separately from independent trainable parameters.

    def logical_resource_summary(self) -> dict:
        conv_apps = 14 if self.topology == "spatial_diag" else 10
        conv_2q = conv_apps * ANSATZ_2Q_PER_APPLICATION[self.ansatz_name]

        pool_apps = len(POOL_STAGE1) + len(POOL_STAGE2) if self.use_pool else 0
        pool_2q = pool_apps * self.pool_2q_per_application

        # Encoding costs depend on the gates used to load the image.
        if self.encoding_name == "reupload":
            enc_2q = N_PATCHES * (N_REUPLOAD_LAYERS - 1)  # four CZs
            enc_note = "4 CZ gates between the two re-upload layers"
        elif self.encoding_name == "e1":
            enc_2q = 16  # 8 * (CNOT-RZ-CNOT)
            enc_note = "16 CNOT gates in the ancilla fusion stage"
        else:
            enc_2q = None
            enc_note = "AmplitudeEmbedding decomposition is backend/transpiler dependent"

        return {
            "independent_trainable_parameters": self.expected_n_params,
            "encoding_parameters": ENCODING_TRAINABLE_PARAMS[self.encoding_name],
            "convolution_parameters": 2 * self.n_conv,
            "pool_parameters": 2 * self.n_pool,
            "convolution_applications_total": conv_apps,
            "pool_applications": pool_apps,
            "logical_2q_convolution": conv_2q,
            "logical_2q_pooling": pool_2q,
            "logical_2q_encoding": enc_2q,
            "logical_2q_encoding_note": enc_note,
            "logical_2q_total_known": (
                enc_2q + conv_2q + pool_2q if enc_2q is not None else None
            ),
        }
