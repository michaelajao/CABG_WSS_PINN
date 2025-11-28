"""
Neural Network Architectures for Physics-Informed Learning

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class FourierFeatures(nn.Module):
    """
    Fourier Feature Encoding for improved high-frequency learning.
    
    Maps input coordinates to higher-dimensional space using sinusoidal functions:
        γ(x) = [sin(2π * B * x), cos(2π * B * x)]
    
    This helps neural networks learn high-frequency patterns that are otherwise
    difficult to capture with standard MLPs (spectral bias problem).
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn High Frequency 
    Functions in Low Dimensional Domains" (NeurIPS 2020)
    """
    
    def __init__(self, in_dim: int = 3, num_frequencies: int = 64, scale: float = 10.0):
        """
        Args:
            in_dim: Input dimension (3 for x, y, z coordinates)
            num_frequencies: Number of random Fourier frequencies
            scale: Standard deviation of random frequency matrix
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
            x: Input coordinates (batch, in_dim)
            
        Returns:
            Encoded features (batch, out_dim)
        """
        # Project to frequency space: (batch, num_frequencies)
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        
        # Concatenate: [original, sin, cos]
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(β * x)
    
    The learnable β parameter allows the network to interpolate between
    linear (β→0) and ReLU-like (β→∞) behavior during training.
    
    Reference: Ramachandran et al., "Searching for Activation Functions" (2017)
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Initial value for learnable scaling parameter
        """
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deep network training.
    
    Architecture: x → Linear → Swish → Linear → Swish → (+x) → output
    
    Skip connections help with gradient flow in deep networks and allow
    the network to learn identity mappings when beneficial.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: Input and output dimension (must be equal for skip connection)
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = Swish()
        
        # Kaiming initialization for layers followed by ReLU-like activations
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc2(self.activation(self.fc1(x))) + x)


class MultiResNetPINN(nn.Module):
    """
    Multi-output ResNet PINN with separate networks for each variable.
    
    This architecture uses independent networks for velocity components (u, v, w),
    pressure (p), and optionally WSS. Each network has its own ResNet backbone,
    allowing independent feature learning at the cost of more parameters.
    
    Architecture (per output):
        Input(3) → Linear(3, hidden) → Swish → [ResBlock] × num_blocks → Linear(hidden, 1)
    
    Total parameters: ~2.6M with default settings (5 networks × ~520K each)
    
    Use this when:
        - Output variables have very different spatial patterns
        - Training time is not critical
        - Memory is not constrained
    """
    
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4,
                 predict_wss: bool = True):
        """
        Initialize the multi-output PINN.
        
        Args:
            hidden_dim: Width of hidden layers in each network
            num_blocks: Number of residual blocks per network
            predict_wss: If True, include WSS prediction network.
                        If False, WSS should be computed from velocity gradients.
        """
        super().__init__()
        
        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Build networks
        def make_network():
            layers = [nn.Linear(3, hidden_dim), Swish()]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Linear(hidden_dim, 1))
            net = nn.Sequential(*layers)
            # Initialize
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            return net
        
        self.net_u = make_network()
        self.net_v = make_network()
        self.net_w = make_network()
        self.net_p = make_network()
        
        if predict_wss:
            self.net_wss = make_network()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaPINN(nn.Module):
    """
    Standard PINN with SiLU activation - simple baseline architecture.
    
    Simple feedforward neural network with SiLU (Swish) activations for
    physics-informed learning. This is the fastest and simplest architecture.
    
    Architecture:
        Input(3) → [Linear → SiLU] × num_blocks → [Output Heads]
    
    SiLU activation: f(x) = x * sigmoid(x)
        - Smooth, non-monotonic
        - Better gradient flow than ReLU
    
    Use this as the baseline PINN architecture for fast experimentation.
    """
    
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4,
                 head_layers: int = 2, predict_wss: bool = True):
        """
        Initialize the standard PINN.
        
        Args:
            hidden_dim: Width of hidden layers
            num_blocks: Number of hidden layers in trunk
            head_layers: Number of layers in each output head (not used, kept for compatibility)
            predict_wss: If True, include WSS prediction head
        """
        super().__init__()
        
        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Build trunk: Input → [Linear → SiLU] × num_blocks
        trunk_layers = []
        trunk_layers.append(nn.Linear(3, hidden_dim))
        trunk_layers.append(nn.SiLU())
        
        for _ in range(num_blocks - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.SiLU())
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Initialize trunk with Xavier initialization (good for SiLU)
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output heads (single linear layer each)
        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_v = nn.Linear(hidden_dim, 1)
        self.head_w = nn.Linear(hidden_dim, 1)
        self.head_p = nn.Linear(hidden_dim, 1)
        
        if predict_wss:
            self.head_wss = nn.Linear(hidden_dim, 1)
        
        # Initialize output heads
        for head in [self.head_u, self.head_v, self.head_w, self.head_p]:
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
        
        if predict_wss:
            nn.init.xavier_normal_(self.head_wss.weight)
            nn.init.zeros_(self.head_wss.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FourierPINN(nn.Module):
    """
    PINN with Fourier Feature Encoding for high-frequency learning.
    
    Fourier features map input coordinates to a higher-dimensional space:
        γ(x) = [x, sin(2πBx), cos(2πBx)]
    
    This helps overcome the spectral bias of standard MLPs, allowing
    the network to learn high-frequency patterns like sharp WSS gradients.
    
    Architecture:
        Input(3) → FourierFeatures → [Linear → SiLU] × num_blocks → [Output Heads]
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (NeurIPS 2020)
    
    Use when:
        - VanillaPINN struggles with sharp spatial gradients
        - WSS prediction shows smoothing artifacts
        - You need better high-frequency detail
    """
    
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4,
                 predict_wss: bool = True, num_frequencies: int = 64,
                 fourier_scale: float = 10.0):
        """
        Initialize the Fourier PINN.
        
        Args:
            hidden_dim: Width of hidden layers
            num_blocks: Number of hidden layers in trunk
            predict_wss: If True, include WSS prediction head
            num_frequencies: Number of random Fourier frequencies
            fourier_scale: Scale of random frequency matrix (higher = more high-freq)
        """
        super().__init__()
        
        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_frequencies = num_frequencies
        self.fourier_scale = fourier_scale
        
        # Fourier feature encoding
        self.fourier = FourierFeatures(
            in_dim=3, 
            num_frequencies=num_frequencies, 
            scale=fourier_scale
        )
        input_dim = self.fourier.out_dim
        
        # Build trunk: FourierFeatures → [Linear → SiLU] × num_blocks
        trunk_layers = []
        trunk_layers.append(nn.Linear(input_dim, hidden_dim))
        trunk_layers.append(nn.SiLU())
        
        for _ in range(num_blocks - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.SiLU())
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Initialize trunk with Xavier initialization (good for SiLU)
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output heads (single linear layer each)
        self.head_u = nn.Linear(hidden_dim, 1)
        self.head_v = nn.Linear(hidden_dim, 1)
        self.head_w = nn.Linear(hidden_dim, 1)
        self.head_p = nn.Linear(hidden_dim, 1)
        
        if predict_wss:
            self.head_wss = nn.Linear(hidden_dim, 1)
        
        # Initialize output heads
        for head in [self.head_u, self.head_v, self.head_w, self.head_p]:
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
        
        if predict_wss:
            nn.init.xavier_normal_(self.head_wss.weight)
            nn.init.zeros_(self.head_wss.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apply Fourier encoding
        x = self.fourier(x)
        
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
    """
    
    def __init__(self, num_splines: int = 8, degree: int = 3, 
                 grid_range: Tuple[float, float] = (-1, 1)):
        """
        Args:
            num_splines: Number of B-spline basis functions
            degree: Degree of B-splines (3 = cubic, most common)
            grid_range: Range of the input domain
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
        
        Args:
            x: Input tensor of shape (...,)
            
        Returns:
            Basis values of shape (..., num_splines)
        """
        # Clamp to grid range
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        
        # Cox-de Boor recursion for B-spline evaluation
        # Start with degree 0 (step functions)
        bases = []
        for i in range(self.num_splines + self.degree):
            left = self.knots[i]
            right = self.knots[i + 1]
            basis = ((x >= left) & (x < right)).float()
            bases.append(basis)
        
        bases = torch.stack(bases, dim=-1)
        
        # Recursively build up to desired degree
        for d in range(1, self.degree + 1):
            new_bases = []
            for i in range(self.num_splines + self.degree - d):
                left_num = x - self.knots[i]
                left_den = self.knots[i + d] - self.knots[i]
                left = torch.where(left_den > 0, left_num / left_den, torch.zeros_like(x))
                
                right_num = self.knots[i + d + 1] - x
                right_den = self.knots[i + d + 1] - self.knots[i + 1]
                right = torch.where(right_den > 0, right_num / right_den, torch.zeros_like(x))
                
                basis = left * bases[..., i] + right * bases[..., i + 1]
                new_bases.append(basis)
            
            bases = torch.stack(new_bases, dim=-1)
        
        return bases


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.
    
    Instead of y = σ(Wx + b) with fixed activation σ,
    KAN uses y_j = Σ_i φ_{ij}(x_i) where each φ_{ij} is a learnable
    B-spline function.
    
    This gives each edge its own learnable activation function,
    providing much more expressivity than standard MLPs.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (-1, 1),
                 base_activation: str = 'silu'):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            grid_size: Number of grid intervals for B-splines
            spline_order: Order of B-splines (3 = cubic)
            grid_range: Range for B-spline grid
            base_activation: Base activation to combine with splines
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
            self.base_act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: Input tensor (batch, in_features)
            
        Returns:
            Output tensor (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Base component: standard linear with activation
        base_output = F.linear(self.base_act(x), self.base_weight * self.base_scale)
        
        # Spline component
        # Evaluate B-splines for each input: (batch, in_features, num_splines)
        spline_basis = self.basis(x)  # (batch, in, num_splines)
        
        # Compute spline activations: (batch, out, in)
        # For each output j and input i: φ_{ji}(x_i) = Σ_k c_{jik} * B_k(x_i)
        spline_output = torch.einsum('bin,oin->bo', spline_basis, 
                                      self.spline_coeffs * self.spline_scale.unsqueeze(-1))
        
        return base_output + spline_output


class KANPINN(nn.Module):
    """
    Kolmogorov-Arnold Network for Physics-Informed Learning.
    
    KAN replaces the fixed activation functions in MLPs with learnable
    B-spline functions on each edge. This allows the network to learn
    the optimal activation shape for each connection.
    
    Key advantages over MLP:
        - Better accuracy with fewer parameters
        - More interpretable (can visualize learned activations)
        - Naturally smooth derivatives (important for PINNs!)
        - 10-100x improvement on some scientific computing tasks
    
    Reference:
        Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756
    
    Note: KAN is computationally more expensive per parameter than MLP,
    but achieves better accuracy with far fewer parameters overall.
    
    Recommended settings:
        - hidden_dim: 32-64 (smaller than MLP!)
        - num_layers: 2-4
        - grid_size: 3-8
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
    ):
        """
        Args:
            in_dim: Input dimension (3 for x,y,z)
            out_dim: Output dimension (5 for u,v,w,p,wss or 4 without wss)
            hidden_dim: Hidden layer width (can be smaller than MLP)
            num_layers: Number of KAN layers
            grid_size: B-spline grid size (more = more expressive)
            spline_order: B-spline order (3 = cubic, recommended)
            predict_wss: If True, include WSS output
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim if predict_wss else out_dim - 1
        self.predict_wss = predict_wss
        
        # Build KAN layers
        layers = []
        
        # Input layer
        layers.append(KANLayer(in_dim, hidden_dim, grid_size, spline_order))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(KANLayer(hidden_dim, hidden_dim, grid_size, spline_order))
        
        # Output layer
        layers.append(KANLayer(hidden_dim, self.out_dim, grid_size, spline_order))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through KAN.
        
        Args:
            x: Input coordinates (batch, 3)
            
        Returns:
            Dictionary with output fields
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
