"""
Neural Network Architectures for Physics-Informed Learning

This module provides two PINN architectures for hemodynamic prediction:

1. SharedTrunkPINN (Recommended):
   - Single shared encoder with multiple output heads
   - More parameter-efficient (~858K params)
   - Faster training due to shared feature computation
   
2. MultiResNetPINN:
   - Separate networks for each output variable
   - More parameters (~2.6M params)
   - May capture independent features better for some problems

Both architectures use:
    - Swish activation with learnable beta parameter
    - ResNet-style skip connections for gradient flow
    - Kaiming initialization for stable training
"""

import torch
import torch.nn as nn
from typing import Dict


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


class SharedTrunkPINN(nn.Module):
    """
    Shared trunk PINN with multiple lightweight output heads (Recommended).
    
    This architecture uses a single shared encoder (trunk) to learn common 
    spatial features, then routes through separate output heads for each 
    variable. This is more efficient than MultiResNetPINN as features are
    computed once and shared.
    
    Architecture:
        Input(3) → [Shared Trunk: Linear + ResBlocks] → Features
                                                         ↓
                                    ┌────────┬────────┬────────┬────────┬────────┐
                                    ↓        ↓        ↓        ↓        ↓
                                  Head_u   Head_v   Head_w   Head_p   Head_wss
                                    ↓        ↓        ↓        ↓        ↓
                                   u(1)    v(1)     w(1)     p(1)    wss(1)
    
    Total parameters: ~858K with default settings (67% fewer than MultiResNetPINN)
    
    Advantages:
        - Faster training (single trunk forward pass)
        - Better parameter efficiency
        - Shared features may improve generalization
    
    Use this as the default choice for most problems.
    """
    
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4,
                 head_layers: int = 2, predict_wss: bool = True):
        """
        Initialize the shared trunk PINN.
        
        Args:
            hidden_dim: Width of trunk and head hidden layers
            num_blocks: Number of residual blocks in shared trunk
            head_layers: Number of layers in each output head
            predict_wss: If True, include WSS prediction head
        """
        super().__init__()
        
        self.predict_wss = predict_wss
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.head_layers = head_layers
        
        # Shared trunk
        trunk_layers = [nn.Linear(3, hidden_dim), Swish()]
        for _ in range(num_blocks):
            trunk_layers.append(ResidualBlock(hidden_dim))
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Initialize trunk
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output heads (smaller networks)
        def make_head():
            layers = []
            for i in range(head_layers):
                if i == head_layers - 1:
                    layers.append(nn.Linear(hidden_dim, 1))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(Swish())
            head = nn.Sequential(*layers)
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            return head
        
        self.head_u = make_head()
        self.head_v = make_head()
        self.head_w = make_head()
        self.head_p = make_head()
        
        if predict_wss:
            self.head_wss = make_head()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Single forward pass through trunk
        features = self.trunk(x)
        
        # Parallel heads
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
