"""
FourierPINN Architecture for Physics-Informed Learning.

This module provides the FourierPINN architecture used in the published paper
for WSS prediction in coronary arteries and saphenous vein grafts.

The FourierPINN uses:
    - Random Fourier feature encoding to overcome spectral bias
    - Residual blocks with SiLU activations for stable gradient flow
    - Learnable Swish activation at the input layer
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONSTANTS
# =============================================================================

INPUT_DIM: int = 3  # x, y, z coordinates
OUTPUT_KEYS: List[str] = ['u', 'v', 'w', 'p', 'wss']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _init_linear_xavier(layer: nn.Linear) -> None:
    """
    Initialize linear layer with Xavier normal and zero bias.

    Xavier initialization is optimal for layers followed by SiLU/Swish
    activations, maintaining variance across layers.
    """
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# =============================================================================
# FOURIER FEATURES
# =============================================================================

class FourierFeatures(nn.Module):
    """
    Fourier Feature Encoding for improved high-frequency learning.

    Maps input coordinates to higher-dimensional space using sinusoidal
    functions:
        gamma(x) = [x, sin(2*pi*B*x), cos(2*pi*B*x)]

    This helps overcome the spectral bias of standard MLPs, allowing
    the network to learn high-frequency patterns like sharp WSS gradients.

    Reference:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency
        Functions in Low Dimensional Domains" (NeurIPS 2020)
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 64,
        scale: float = 10.0
    ) -> None:
        """
        Initialize Fourier feature encoding.

        Args:
            in_dim: Input dimension (3 for x, y, z coordinates).
            num_frequencies: Number of random Fourier frequencies.
            scale: Standard deviation of random frequency matrix (sigma).
                Higher values capture more high-frequency content.
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.out_dim = in_dim + 2 * num_frequencies  # original + sin + cos

        # Random frequency matrix (fixed, not learned)
        B = torch.randn(num_frequencies, in_dim) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.

        Args:
            x: Input coordinates with shape (batch, in_dim).

        Returns:
            Encoded features with shape (batch, out_dim).
        """
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

class Swish(nn.Module):
    """
    Learnable Swish activation function.

    Swish: f(x) = x * sigmoid(beta * x)

    When beta=1, this is equivalent to SiLU. The learnable beta allows
    the network to adapt the activation shape during training.

    Reference:
        Ramachandran et al., "Searching for Activation Functions" (2017)
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation: x * sigmoid(beta * x)."""
        return x * torch.sigmoid(self.beta * x)


# =============================================================================
# RESIDUAL BLOCKS
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with SiLU activation for deep network training.

    Architecture: x -> Linear -> SiLU -> Linear -> SiLU -> (+x) -> output

    SiLU (Sigmoid Linear Unit) provides smooth, infinitely differentiable
    activations that are essential for computing physics residuals via
    automatic differentiation in PINNs.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize residual block with SiLU activation.

        Args:
            dim: Input and output dimension (must be equal for skip connection).
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        _init_linear_xavier(self.fc1)
        _init_linear_xavier(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x) + residual)
        return x


# =============================================================================
# FOURIER PINN (MAIN ARCHITECTURE)
# =============================================================================

class FourierPINN(nn.Module):
    """
    PINN with Fourier Feature Encoding and Learnable Swish Activation.

    This is the architecture used in the published paper. It combines:
        - Fourier feature encoding for high-frequency learning
        - ResNet skip connections for stable gradient flow
        - Learnable Swish activation at the input layer

    Architecture:
        Input(3) -> FourierFeatures -> Linear -> Swish(beta)
        -> [ResBlock] x num_blocks -> [Output Heads]

    Each ResBlock: x -> Linear -> SiLU -> Linear -> (+x) -> SiLU

    Paper configuration:
        - hidden_dim: 48
        - num_blocks: 6
        - num_frequencies: 64
        - fourier_scale: 10.0
        - Total parameters: ~34,000

    Reference:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency
        Functions in Low Dimensional Domains" (NeurIPS 2020)
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        num_blocks: int = 6,
        predict_wss: bool = True,
        num_frequencies: int = 64,
        fourier_scale: float = 10.0
    ) -> None:
        """
        Initialize the Fourier PINN with ResNet architecture.

        Args:
            hidden_dim: Width of hidden layers (paper uses 48).
            num_blocks: Number of ResNet blocks (paper uses 6).
            predict_wss: If True, include WSS prediction head.
            num_frequencies: Number of random Fourier frequencies (paper uses 64).
            fourier_scale: Scale of random frequency matrix sigma (paper uses 10.0).
        """
        super().__init__()

        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_frequencies = num_frequencies
        self.fourier_scale = fourier_scale

        # Fourier feature encoding
        self.fourier = FourierFeatures(
            in_dim=INPUT_DIM,
            num_frequencies=num_frequencies,
            scale=fourier_scale
        )
        input_dim = self.fourier.out_dim

        # Input projection with learnable Swish
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = Swish(beta=1.0)
        _init_linear_xavier(self.input_layer)

        # ResNet blocks with skip connections
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output heads
        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_v = nn.Linear(hidden_dim, 1)
        self.head_w = nn.Linear(hidden_dim, 1)
        self.head_p = nn.Linear(hidden_dim, 1)

        if predict_wss:
            self.head_wss = nn.Linear(hidden_dim, 1)

        # Initialize output heads
        for head in [self.head_u, self.head_v, self.head_w, self.head_p]:
            _init_linear_xavier(head)

        if predict_wss:
            _init_linear_xavier(self.head_wss)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input coordinates with shape (batch, 3).

        Returns:
            Dictionary with keys 'u', 'v', 'w', 'p' and optionally 'wss',
            each mapping to a tensor of shape (batch, 1).
        """
        # Fourier encoding
        x = self.fourier(x)

        # Input projection with learnable Swish
        features = self.input_layer(x)
        features = self.input_activation(features)

        # ResNet blocks with skip connections
        for block in self.blocks:
            features = block(features)

        # Output predictions
        outputs = {
            'u': self.head_u(features),
            'v': self.head_v(features),
            'w': self.head_w(features),
            'p': self.head_p(features),
        }
        if self.predict_wss:
            outputs['wss'] = self.head_wss(features)
        return outputs

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
