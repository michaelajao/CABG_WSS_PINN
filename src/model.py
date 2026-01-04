"""
Neural Network Architectures for Physics-Informed Learning.

This module provides PINN architectures for hemodynamic prediction:

1. VanillaPINN:
   - Standard MLP with SiLU activation
   - Simple and fast baseline

2. FourierPINN:
   - Fourier feature encoding for high-frequency learning
   - Better for sharp WSS gradients

3. MultiResNetPINN:
   - Separate networks for each output variable
   - ResNet blocks with skip connections
   - More parameters but independent feature learning

4. KANPINN (Experimental):
   - Kolmogorov-Arnold Networks with learnable B-spline activations
   - Better accuracy with fewer parameters
   - Naturally smooth derivatives (beneficial for PINNs)

All architectures use careful initialization for stable training.

Attributes:
    INPUT_DIM (int): Default input dimension (3 for x, y, z coordinates).
    OUTPUT_KEYS (list): Standard output field names ['u', 'v', 'w', 'p', 'wss'].

Classes:
    FourierFeatures: Fourier feature encoding layer.
    Swish: Learnable Swish activation function.
    ResidualBlock: Residual block with skip connections.
    VanillaPINN: Standard MLP-based PINN.
    FourierPINN: PINN with Fourier feature encoding.
    MultiResNetPINN: Multi-output ResNet PINN.
    BSplineBasis: B-spline basis functions for KAN.
    KANLayer: Kolmogorov-Arnold Network layer.
    KANPINN: KAN-based PINN (experimental).
"""

from typing import Dict, List, Tuple

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

    Args:
        layer: Linear layer to initialize.
    """
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_linear_kaiming(layer: nn.Linear) -> None:
    """
    Initialize linear layer with Kaiming normal and zero bias.

    Kaiming initialization is optimal for layers followed by ReLU-like
    activations (ReLU, LeakyReLU, etc.).

    Args:
        layer: Linear layer to initialize.
    """
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _init_module_weights(
    module: nn.Module,
    init_fn: callable = _init_linear_xavier
) -> None:
    """
    Initialize all Linear layers in a module.

    Args:
        module: PyTorch module containing Linear layers.
        init_fn: Initialization function to apply to each Linear layer.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_fn(m)


# =============================================================================
# FOURIER FEATURES
# =============================================================================

class FourierFeatures(nn.Module):
    """
    Fourier Feature Encoding for improved high-frequency learning.

    Maps input coordinates to higher-dimensional space using sinusoidal
    functions:
        gamma(x) = [x, sin(2*pi*B*x), cos(2*pi*B*x)]

    This helps neural networks learn high-frequency patterns that are
    otherwise difficult to capture with standard MLPs (spectral bias).

    Attributes:
        in_dim (int): Input dimension.
        num_frequencies (int): Number of random Fourier frequencies.
        out_dim (int): Output dimension (in_dim + 2 * num_frequencies).
        B (Tensor): Random frequency matrix (buffer, not learned).

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
            scale: Standard deviation of random frequency matrix.
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
        # Project to frequency space: (batch, num_frequencies)
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)

        # Concatenate: [original, sin, cos]
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

class Swish(nn.Module):
    """
    Learnable Swish activation function.

    Swish: f(x) = x * sigmoid(beta * x)

    When beta=1, this is equivalent to SiLU. The learnable beta allows
    the network to adapt the activation shape during training, which can
    improve convergence for PINNs.

    Attributes:
        beta (Parameter): Learnable scaling parameter, initialized to 1.0.

    Reference:
        Ramachandran et al., "Searching for Activation Functions" (2017)
    """

    def __init__(self, beta: float = 1.0) -> None:
        """
        Initialize learnable Swish activation.

        Args:
            beta: Initial value for the learnable beta parameter.
        """
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

    Note:
        Input and output dimensions must be equal for the skip connection.

    Attributes:
        fc1 (Linear): First linear layer.
        fc2 (Linear): Second linear layer.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize residual block with SiLU activation.

        Args:
            dim: Input and output dimension (must be equal).
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        # Xavier initialization (optimal for SiLU)
        _init_linear_xavier(self.fc1)
        _init_linear_xavier(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor with shape (batch, dim).

        Returns:
            Output tensor with shape (batch, dim).
        """
        residual = x
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x) + residual)
        return x


class ResidualBlockLN(nn.Module):
    """
    Residual block with LayerNorm and learnable Swish activation.

    Architecture (Pre-norm style):
        x -> LayerNorm -> Linear -> Swish -> LayerNorm -> Linear -> (+x) -> Swish

    LayerNorm stabilizes training by normalizing activations, while
    learnable Swish adapts the activation shape during training.

    This is the recommended block for FourierPINN when training stability
    or deeper networks are needed.

    Attributes:
        ln1 (LayerNorm): First layer normalization.
        fc1 (Linear): First linear layer.
        swish1 (Swish): First learnable Swish activation.
        ln2 (LayerNorm): Second layer normalization.
        fc2 (Linear): Second linear layer.
        swish2 (Swish): Second learnable Swish activation.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize residual block with LayerNorm and learnable Swish.

        Args:
            dim: Input and output dimension (must be equal).
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.swish1 = Swish(beta=1.0)

        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.swish2 = Swish(beta=1.0)

        # Xavier initialization (optimal for Swish)
        _init_linear_xavier(self.fc1)
        _init_linear_xavier(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connection.

        Args:
            x: Input tensor with shape (batch, dim).

        Returns:
            Output tensor with shape (batch, dim).
        """
        residual = x
        x = self.swish1(self.fc1(self.ln1(x)))
        x = self.swish2(self.fc2(self.ln2(x)) + residual)
        return x


# =============================================================================
# PINN ARCHITECTURES
# =============================================================================

class VanillaPINN(nn.Module):
    """
    Standard PINN with SiLU activation - simple baseline architecture.

    Simple feedforward neural network with SiLU (Swish) activations for
    physics-informed learning. This is the fastest and simplest architecture.

    Architecture:
        Input(3) -> [Linear -> SiLU] x num_blocks -> [Output Heads]

    SiLU activation: f(x) = x * sigmoid(x)
        - Smooth, non-monotonic
        - Better gradient flow than ReLU

    Use this as the baseline PINN architecture for fast experimentation.

    Attributes:
        predict_wss (bool): Whether WSS output is included.
        hidden_dim (int): Width of hidden layers.
        num_blocks (int): Number of hidden layers.
        trunk (Sequential): Shared feature extraction layers.
        head_u, head_v, head_w, head_p (Linear): Velocity and pressure heads.
        head_wss (Linear): WSS head (if predict_wss=True).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        predict_wss: bool = True
    ) -> None:
        """
        Initialize the standard PINN.

        Args:
            hidden_dim: Width of hidden layers.
            num_blocks: Number of hidden layers in trunk.
            predict_wss: If True, include WSS prediction head.
        """
        super().__init__()

        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Shared trunk: Input -> [Linear -> SiLU] x num_blocks
        trunk_layers: List[nn.Module] = []
        trunk_layers.append(nn.Linear(INPUT_DIM, hidden_dim))
        trunk_layers.append(nn.SiLU())

        for _ in range(num_blocks - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.SiLU())

        self.trunk = nn.Sequential(*trunk_layers)

        # Initialize trunk with Xavier (optimal for SiLU)
        _init_module_weights(self.trunk, _init_linear_xavier)

        # Output heads (single linear layer each)
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
        # Forward pass through trunk
        features = self.trunk(x)

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
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FourierPINN(nn.Module):
    """
    PINN with Fourier Feature Encoding and Learnable Swish Activation.

    Combines Fourier feature encoding for high-frequency learning with
    ResNet skip connections and learnable Swish activations.

    Fourier features map input coordinates to a higher-dimensional space:
        gamma(x) = [x, sin(2*pi*B*x), cos(2*pi*B*x)]

    This helps overcome the spectral bias of standard MLPs, allowing
    the network to learn high-frequency patterns like sharp WSS gradients.

    Architecture:
        Input(3) -> FourierFeatures -> Linear -> Swish(beta)
        -> [ResBlock] x num_blocks -> [Output Heads]

    Each ResBlock: x -> Linear -> SiLU -> Linear -> (+x) -> SiLU

    Features:
        - Fourier features for high-frequency learning
        - Learnable Swish at input (beta parameter adapts during training)
        - ResNet blocks for stable gradient flow
        - No LayerNorm (preserves physics gradient information)

    Attributes:
        predict_wss (bool): Whether WSS output is included.
        hidden_dim (int): Width of hidden layers.
        num_blocks (int): Number of ResNet blocks.
        num_frequencies (int): Number of Fourier frequencies.
        fourier_scale (float): Scale of frequency matrix.
        fourier (FourierFeatures): Fourier encoding layer.
        input_layer (Linear): Projects Fourier features to hidden dim.
        input_activation (Swish): Learnable Swish activation.
        blocks (ModuleList): List of ResidualBlock modules.
        head_u, head_v, head_w, head_p (Linear): Velocity and pressure heads.
        head_wss (Linear): WSS head (if predict_wss=True).

    Reference:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency
        Functions in Low Dimensional Domains" (NeurIPS 2020)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        predict_wss: bool = True,
        num_frequencies: int = 64,
        fourier_scale: float = 10.0
    ) -> None:
        """
        Initialize the Fourier PINN with ResNet architecture.

        Args:
            hidden_dim: Width of hidden layers.
            num_blocks: Number of ResNet blocks in trunk.
            predict_wss: If True, include WSS prediction head.
            num_frequencies: Number of random Fourier frequencies.
            fourier_scale: Scale of random frequency matrix. Higher values
                capture more high-frequency content.
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

        # ResNet blocks with skip connections (fast, no LayerNorm)
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
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PIRATENET - Physics-Informed Residual AdapTivE Networks
# =============================================================================

class PirateNetBlock(nn.Module):
    """
    PirateNet residual block with gating and adaptive skip connections.

    Each block consists of 3 dense layers with 2 gating operations,
    followed by an adaptive residual connection.

    Forward pass:
        f = σ(W₁x + b₁)
        z₁ = f ⊙ U + (1-f) ⊙ V      # gating
        g = σ(W₂z₁ + b₂)
        z₂ = g ⊙ U + (1-g) ⊙ V      # gating
        h = σ(W₃z₂ + b₃)
        output = α·h + (1-α)·x       # adaptive skip

    Key innovation: α is initialized to 0, so the network starts as
    an identity mapping and gradually learns nonlinearity.

    Attributes:
        fc1, fc2, fc3 (Linear): Three dense layers.
        alpha (Parameter): Adaptive skip connection weight, initialized to 0.

    Reference:
        Wang et al., "PirateNets: Physics-informed Deep Learning with
        Residual Adaptive Networks" (2024)
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize PirateNet block.

        Args:
            dim: Hidden dimension (input and output must be equal).
        """
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

        # Adaptive skip connection weight
        # Per original PirateNet paper: α=0 means network starts as identity mapping
        # This avoids initialization pathology and allows gradual learning of nonlinearity
        # α will increase during training as the network learns
        self.alpha = nn.Parameter(torch.tensor([0.0]))

        # Initialize weights with Xavier (Glorot), biases to zero
        _init_linear_xavier(self.fc1)
        _init_linear_xavier(self.fc2)
        _init_linear_xavier(self.fc3)

    def forward(
        self,
        x: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through PirateNet block with gating.

        Args:
            x: Input tensor with shape (batch, dim).
            U: First encoding gate with shape (batch, dim).
            V: Second encoding gate with shape (batch, dim).

        Returns:
            Output tensor with shape (batch, dim).
        """
        # First layer + gating
        f = torch.tanh(self.fc1(x))
        z1 = f * U + (1 - f) * V

        # Second layer + gating
        g = torch.tanh(self.fc2(z1))
        z2 = g * U + (1 - g) * V

        # Third layer
        h = torch.tanh(self.fc3(z2))

        # Adaptive skip connection: α·h + (1-α)·x
        # When α=0 (initialization): output = x (identity)
        # As α increases: more nonlinearity is added
        return self.alpha * h + (1 - self.alpha) * x


class PirateNetPINN(nn.Module):
    """
    Physics-Informed Residual AdapTivE Networks (PirateNets).

    PirateNets address initialization pathologies in PINNs through:
    1. Random Fourier feature encoding for high-frequency learning
    2. Encoding gates (U, V) that modulate information flow
    3. Adaptive skip connections with α initialized to 0

    The α=0 initialization means the network starts as an identity
    mapping, avoiding the problematic random initialization that
    can cause PINNs to get stuck in poor local minima.

    Architecture:
        Input(3) -> FourierFeatures -> [U, V encoders]
        -> [PirateNetBlock × num_blocks] -> Output Heads

    Use when:
        - Standard PINNs struggle to converge
        - Physics losses remain high despite long training
        - You need better initialization for complex PDEs

    Attributes:
        predict_wss (bool): Whether WSS output is included.
        hidden_dim (int): Width of hidden layers.
        num_blocks (int): Number of PirateNet blocks.
        num_frequencies (int): Number of Fourier frequencies.
        fourier_scale (float): Scale of frequency matrix.

    Reference:
        Wang et al., "PirateNets: Physics-informed Deep Learning with
        Residual Adaptive Networks" (2024)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        predict_wss: bool = True,
        num_frequencies: int = 64,
        fourier_scale: float = 10.0
    ) -> None:
        """
        Initialize PirateNet PINN.

        Args:
            hidden_dim: Width of hidden layers.
            num_blocks: Number of PirateNet blocks.
            predict_wss: If True, include WSS prediction head.
            num_frequencies: Number of random Fourier frequencies.
            fourier_scale: Scale of random frequency matrix.
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
        fourier_dim = self.fourier.out_dim

        # Encoding gates U and V (computed once from input)
        # These modulate information flow through all blocks
        self.encoder_U = nn.Linear(fourier_dim, hidden_dim)
        self.encoder_V = nn.Linear(fourier_dim, hidden_dim)
        _init_linear_xavier(self.encoder_U)
        _init_linear_xavier(self.encoder_V)

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(fourier_dim, hidden_dim)
        _init_linear_xavier(self.input_proj)

        # PirateNet blocks with gating and adaptive skip connections
        self.blocks = nn.ModuleList([
            PirateNetBlock(hidden_dim) for _ in range(num_blocks)
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
        Forward pass through PirateNet.

        Args:
            x: Input coordinates with shape (batch, 3).

        Returns:
            Dictionary with keys 'u', 'v', 'w', 'p' and optionally 'wss',
            each mapping to a tensor of shape (batch, 1).
        """
        # Fourier encoding
        phi = self.fourier(x)

        # Compute encoding gates (used by all blocks)
        U = torch.tanh(self.encoder_U(phi))
        V = torch.tanh(self.encoder_V(phi))

        # Project input to hidden dimension
        features = torch.tanh(self.input_proj(phi))

        # Pass through PirateNet blocks
        for block in self.blocks:
            features = block(features, U, V)

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

    def get_alpha_values(self) -> List[float]:
        """
        Get current alpha values from all blocks.

        Useful for monitoring how the network transitions from
        identity mapping to nonlinear transformation.

        Returns:
            List of alpha values, one per block.
        """
        return [block.alpha.item() for block in self.blocks]


class MultiResNetPINN(nn.Module):
    """
    Multi-output ResNet PINN with separate networks for each variable.

    This architecture uses independent networks for velocity components
    (u, v, w), pressure (p), and WSS. Each network has its own ResNet
    backbone, allowing independent feature learning at the cost of more
    parameters.

    WARNING: This architecture breaks the physics coupling between velocity
    components. In incompressible flow, u, v, w are coupled through the
    continuity equation (div(u) = 0). Having independent networks means
    each learns features separately, which may result in:
    - Higher continuity residuals
    - Less physically consistent solutions
    - Worse generalization for physics-informed training

    Consider using FourierPINN or VanillaPINN for physics-informed training,
    as their shared trunks naturally learn coupled representations.

    Architecture (per output):
        Input(3) -> Linear(3, hidden) -> SiLU -> [ResBlock] x num_blocks
        -> Linear(hidden, 1)

    Use this when:
        - Output variables have very different spatial patterns
        - Pure data-driven training (no physics constraints)
        - Experimental comparison with shared-trunk architectures

    Attributes:
        predict_wss (bool): Whether WSS output is included.
        hidden_dim (int): Width of hidden layers in each network.
        num_blocks (int): Number of residual blocks per network.
        net_u, net_v, net_w, net_p (Sequential): Individual networks.
        net_wss (Sequential): WSS network (if predict_wss=True).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        predict_wss: bool = True
    ) -> None:
        """
        Initialize the multi-output PINN.

        Args:
            hidden_dim: Width of hidden layers in each network.
            num_blocks: Number of residual blocks per network.
            predict_wss: If True, include WSS prediction head.
        """
        super().__init__()

        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Build networks
        def make_network() -> nn.Sequential:
            layers: List[nn.Module] = [nn.Linear(INPUT_DIM, hidden_dim), nn.SiLU()]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Linear(hidden_dim, 1))
            net = nn.Sequential(*layers)
            _init_module_weights(net, _init_linear_xavier)
            return net

        self.net_u = make_network()
        self.net_v = make_network()
        self.net_w = make_network()
        self.net_p = make_network()

        if predict_wss:
            self.net_wss = make_network()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all networks.

        Args:
            x: Input coordinates with shape (batch, 3).

        Returns:
            Dictionary with keys 'u', 'v', 'w', 'p' and optionally 'wss',
            each mapping to a tensor of shape (batch, 1).
        """
        outputs = {
            'u': self.net_u(x),
            'v': self.net_v(x),
            'w': self.net_w(x),
            'p': self.net_p(x),
        }
        if self.predict_wss:
            outputs['wss'] = self.net_wss(x)
        return outputs

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# KOLMOGOROV-ARNOLD NETWORK (KAN) - Liu et al. 2024
# =============================================================================

class BSplineBasis(nn.Module):
    """
    B-spline basis functions for KAN.

    B-splines provide smooth, local basis functions that can approximate
    any continuous function. The learnable coefficients determine the
    shape of the activation function.

    Attributes:
        num_splines (int): Number of B-spline basis functions.
        degree (int): Degree of B-splines.
        grid_range (tuple): Range of the input domain.
        knots (Tensor): Uniform knot vector (buffer).
    """

    def __init__(
        self,
        num_splines: int = 8,
        degree: int = 3,
        grid_range: Tuple[float, float] = (-1, 1)
    ) -> None:
        """
        Initialize B-spline basis.

        Args:
            num_splines: Number of B-spline basis functions.
            degree: Degree of B-splines (3 = cubic, most common).
            grid_range: Range of the input domain.
        """
        super().__init__()
        self.num_splines = num_splines
        self.degree = degree
        self.grid_range = grid_range

        # Create uniform knot vector
        num_knots = num_splines + degree + 1
        knots = torch.linspace(grid_range[0], grid_range[1], num_knots)
        self.register_buffer('knots', knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at x.

        Uses vectorized Cox-de Boor recursion for B-spline evaluation.

        Args:
            x: Input tensor of shape (...,).

        Returns:
            Basis values of shape (..., num_splines).
        """
        # Clamp to grid range
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])

        # Vectorized Cox-de Boor recursion: start with degree 0 (step functions)
        # Instead of loop, use broadcasting
        num_intervals = self.num_splines + self.degree
        left_knots = self.knots[:num_intervals]    # Shape: (num_intervals,)
        right_knots = self.knots[1:num_intervals+1]  # Shape: (num_intervals,)

        # Expand x for broadcasting: (..., 1)
        x_expanded = x.unsqueeze(-1)

        # Compute all degree-0 bases at once: (..., num_intervals)
        bases = ((x_expanded >= left_knots) & (x_expanded < right_knots)).float()

        # Recursively build up to desired degree (still needs loop over degrees)
        for d in range(1, self.degree + 1):
            num_bases = self.num_splines + self.degree - d

            # Vectorized computation for this degree level
            # Indices for current level
            knots_i = self.knots[:num_bases]
            knots_i_d = self.knots[d:d+num_bases]
            knots_i_d_1 = self.knots[d+1:d+1+num_bases]
            knots_i_1 = self.knots[1:1+num_bases]

            # Left term: (x - t_i) / (t_{i+d} - t_i)
            left_num = x_expanded - knots_i
            left_den = knots_i_d - knots_i
            left_den_safe = torch.where(left_den > 0, left_den, torch.ones_like(left_den))
            left = torch.where(
                left_den > 0,
                left_num / left_den_safe,
                torch.zeros_like(left_num)
            )

            # Right term: (t_{i+d+1} - x) / (t_{i+d+1} - t_{i+1})
            right_num = knots_i_d_1 - x_expanded
            right_den = knots_i_d_1 - knots_i_1
            right_den_safe = torch.where(right_den > 0, right_den, torch.ones_like(right_den))
            right = torch.where(
                right_den > 0,
                right_num / right_den_safe,
                torch.zeros_like(right_num)
            )

            # Combine: B_{i,d}(x) = left * B_{i,d-1}(x) + right * B_{i+1,d-1}(x)
            bases = left * bases[..., :num_bases] + right * bases[..., 1:num_bases+1]

        return bases


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.

    Instead of y = activation(Wx + b) with fixed activation,
    KAN uses y_j = sum_i phi_{ij}(x_i) where each phi_{ij} is a learnable
    B-spline function.

    This gives each edge its own learnable activation function,
    providing much more expressivity than standard MLPs.

    Attributes:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        grid_size (int): Number of grid intervals for B-splines.
        spline_order (int): Order of B-splines.
        spline_coeffs (Parameter): Learnable B-spline coefficients.
        base_weight (Parameter): Base weight for residual connection.
        spline_scale (Parameter): Scale for spline component.
        base_scale (Parameter): Scale for base component.
        basis (BSplineBasis): B-spline basis functions.
        base_act (Module): Base activation function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1, 1),
        base_activation: str = 'silu'
    ) -> None:
        """
        Initialize KAN layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            grid_size: Number of grid intervals for B-splines.
            spline_order: Order of B-splines (3 = cubic).
            grid_range: Range for B-spline grid.
            base_activation: Base activation to combine with splines.
                Supported: 'silu', 'gelu'.

        Raises:
            ValueError: If base_activation is not supported.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Number of B-spline coefficients per edge
        num_splines = grid_size + spline_order

        # Learnable B-spline coefficients: (out, in, num_splines)
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, num_splines) * 0.1
        )

        # Base weight (like standard linear layer, for residual)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        )

        # Scale parameters for combining base and spline
        self.spline_scale = nn.Parameter(torch.ones(out_features, in_features))
        self.base_scale = nn.Parameter(torch.ones(out_features, in_features))

        # B-spline basis
        self.basis = BSplineBasis(num_splines, spline_order, grid_range)

        # Base activation
        if base_activation == 'silu':
            self.base_act = nn.SiLU()
        elif base_activation == 'gelu':
            self.base_act = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation: {base_activation}. "
                f"Supported: 'silu', 'gelu'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.

        Args:
            x: Input tensor with shape (batch, in_features).

        Returns:
            Output tensor with shape (batch, out_features).
        """
        # Base component: standard linear with activation
        base_output = F.linear(
            self.base_act(x),
            self.base_weight * self.base_scale
        )

        # Spline component
        # Evaluate B-splines for each input: (batch, in_features, num_splines)
        spline_basis = self.basis(x)  # (batch, in, num_splines)

        # Compute spline activations: (batch, out, in)
        # For each output j and input i: phi_{ji}(x_i) = sum_k c_{jik} * B_k(x_i)
        spline_output = torch.einsum(
            'bin,oin->bo',
            spline_basis,
            self.spline_coeffs * self.spline_scale.unsqueeze(-1)
        )

        return base_output + spline_output


class KANPINN(nn.Module):
    """
    Kolmogorov-Arnold Network for Physics-Informed Learning (Experimental).

    KAN replaces the fixed activation functions in MLPs with learnable
    B-spline functions on each edge. This allows the network to learn
    the optimal activation shape for each connection.

    Key characteristics:
        - Learnable activation functions per edge
        - More interpretable (can visualize learned activations)
        - Naturally smooth derivatives (beneficial for PINNs)
        - Higher computational cost per parameter than MLP

    Recommended settings:
        - hidden_dim: 32-64
        - num_layers: 2-4
        - grid_size: 3-8

    Attributes:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        predict_wss (bool): Whether WSS output is included.
        layers (ModuleList): List of KAN layers.

    Reference:
        Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks.
        arXiv:2404.19756
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        predict_wss: bool = True
    ) -> None:
        """
        Initialize KAN PINN.

        Args:
            in_dim: Input dimension (3 for x, y, z).
            out_dim: Output dimension (5 for u, v, w, p, wss or 4 without wss).
            hidden_dim: Hidden layer width (can be smaller than MLP).
            num_layers: Number of KAN layers.
            grid_size: B-spline grid size (more = more expressive).
            spline_order: B-spline order (3 = cubic, recommended).
            predict_wss: If True, include WSS output.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim if predict_wss else out_dim - 1
        self.predict_wss = predict_wss

        # Build KAN layers
        layers: List[KANLayer] = []

        # Input layer
        layers.append(KANLayer(in_dim, hidden_dim, grid_size, spline_order))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                KANLayer(hidden_dim, hidden_dim, grid_size, spline_order)
            )

        # Output layer
        layers.append(
            KANLayer(hidden_dim, self.out_dim, grid_size, spline_order)
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through KAN.

        Args:
            x: Input coordinates with shape (batch, 3).

        Returns:
            Dictionary with keys 'u', 'v', 'w', 'p' and optionally 'wss',
            each mapping to a tensor of shape (batch, 1).
        """
        for layer in self.layers:
            x = layer(x)

        outputs = {
            'u': x[:, 0:1],
            'v': x[:, 1:2],
            'w': x[:, 2:3],
            'p': x[:, 3:4],
        }

        if self.predict_wss:
            outputs['wss'] = x[:, 4:5]

        return outputs

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


