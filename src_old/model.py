"""
PINN model architecture

Contains the Physics-Informed Neural Network model definition and utilities
for gradient computation required for physics losses.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .config import PINN_LAYERS


def initialize_weights(model: nn.Module, method: str = 'xavier_normal'):
    """
    Initialize model weights

    Args:
        model: PyTorch model
        method: Initialization method ('xavier_normal', 'xavier_uniform',
                'kaiming_normal', 'kaiming_uniform')
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class LearnableSwish(nn.Module):
    """Swish activation with learnable beta: x * sigmoid(beta * x)."""

    def __init__(self, beta: float = 1.0, trainable: bool = True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for hemodynamics

    The network predicts fluid flow variables from spatial coordinates:
    - Input: (x, y, z) coordinates
    - Output: velocity field (u, v, w) + pressure (p) + wall shear stress (wss)

    The network is trained with physics constraints from Navier-Stokes and
    continuity equations.
    """

    def __init__(self, layers: List[int] = PINN_LAYERS, activation: str = 'tanh', activation_beta: Optional[float] = None):
        """
        Initialize PINN model

        Args:
            layers: List of layer sizes [input, hidden1, hidden2, ..., output]
                   Default: [3, 128, 256, 512, 256, 128, 5]
                   Output: [u, v, w, p, wss] (5 neurons)
        """
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Activation function
        act = activation.lower() if isinstance(activation, str) else 'tanh'
        if act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'gelu':
            self.activation = nn.GELU()
        elif act in ('silu',):
            self.activation = nn.SiLU()
        elif act in ('swish', 'swish_learnable', 'lswish'):
            # If activation_beta provided, create learnable Swish
            beta = 1.0 if activation_beta is None else float(activation_beta)
            self.activation = LearnableSwish(beta=beta, trainable=True)
        elif act == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_name = act
        self.activation_beta = float(activation_beta) if activation_beta is not None else None

        # Output layer splits
        self.output_split = [3, 1, 1]  # [velocity(u,v,w), pressure, wss]

        # Store layer configuration
        self.layer_sizes = layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            x: Input coordinates (batch_size, 3) - [x, y, z]

        Returns:
            Dictionary with output variables:
                - 'velocity': velocity vector (batch_size, 3) - [u, v, w]
                - 'u': x-component of velocity (batch_size, 1)
                - 'v': y-component of velocity (batch_size, 1)
                - 'w': z-component of velocity (batch_size, 1)
                - 'pressure': pressure field (batch_size, 1)
                - 'wss': wall shear stress (batch_size, 1)
        """
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))

        # Final layer (no activation for outputs)
        x = self.layers[-1](x)

        # Split outputs
        u = x[:, 0:1]  # velocity x-component
        v = x[:, 1:2]  # velocity y-component
        w = x[:, 2:3]  # velocity z-component
        p = x[:, 3:4]  # pressure
        wss = x[:, 4:5]  # wall shear stress

        return {
            'velocity': torch.cat([u, v, w], dim=1),
            'u': u,
            'v': v,
            'w': w,
            'pressure': p,
            'wss': wss
        }

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params
        }

    def print_architecture(self):
        """Print model architecture summary"""
        print("="*80)
        if self.activation_name in ('swish', 'swish_learnable', 'lswish') and hasattr(self.activation, 'beta'):
            beta_val = float(self.activation.beta.detach().cpu().item())
            print(f"Architecture: PINN | Activation: {self.activation_name.upper()} (β={beta_val:.4f}, learnable)")
        else:
            print(f"Architecture: PINN | Activation: {self.activation_name.upper()}")
        print(f"Layer Configuration: {self.layer_sizes}")
        print("="*80)

        param_info = self.count_parameters()
        print(f"Total Parameters: {param_info['total']:,} (all trainable)")
        print(f"Outputs: [u, v, w, p, wss] -> Velocity (m/s), Pressure (Pa), WSS (Pa)")
        print("="*80)


def compute_gradients(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    grad_outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute gradients of outputs with respect to inputs

    This function is essential for computing physics-informed losses that
    require derivatives of network outputs.

    Args:
        outputs: Network outputs to differentiate
        inputs: Input variables (must have requires_grad=True)
        grad_outputs: Gradient of some scalar with respect to outputs
                     If None, assumes outputs are scalar or uses ones

    Returns:
        Gradients of outputs with respect to inputs
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(outputs)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,  # Allow higher-order derivatives
        retain_graph=True,  # Keep computation graph for multiple grad calls
        only_inputs=True    # Only compute gradients w.r.t. inputs
    )[0]

    return grads


class ResBlock(nn.Module):
    """Simple pre-activation ResNet block for MLPs."""

    def __init__(self, width: int, activation: nn.Module):
        super().__init__()
        self.act = activation
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(x)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return residual + out


class ResNetPINN(nn.Module):
    """ResNet-style PINN with residual blocks for stable deep MLPs.

    Input: (x, y, z)
    Output: [u, v, w, p, wss] (5)
    """

    def __init__(
        self,
        width: int = 256,
        blocks: int = 6,
        activation: str = 'tanh',
        activation_beta: Optional[float] = None,
    ):
        super().__init__()

        # Build activation module (reuse PINN choices)
        act_name = activation.lower() if isinstance(activation, str) else 'tanh'
        if act_name == 'tanh':
            act = nn.Tanh()
        elif act_name == 'relu':
            act = nn.ReLU()
        elif act_name == 'gelu':
            act = nn.GELU()
        elif act_name in ('silu',):
            act = nn.SiLU()
        elif act_name in ('swish', 'swish_learnable', 'lswish'):
            beta = 1.0 if activation_beta is None else float(activation_beta)
            act = LearnableSwish(beta=beta, trainable=True)
        elif act_name == 'leaky_relu':
            act = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation_name = act_name
        self.activation = act
        self.width = width
        self.blocks = blocks

        # Stem and head
        self.in_linear = nn.Linear(3, width)
        self.resblocks = nn.ModuleList([ResBlock(width, self.activation) for _ in range(blocks)])
        self.out_linear = nn.Linear(width, 5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.in_linear(x)
        for b in self.resblocks:
            h = b(h)
        h = self.activation(h)
        x = self.out_linear(h)

        u = x[:, 0:1]
        v = x[:, 1:2]
        w = x[:, 2:3]
        p = x[:, 3:4]
        wss = x[:, 4:5]

        return {
            'velocity': torch.cat([u, v, w], dim=1),
            'u': u,
            'v': v,
            'w': w,
            'pressure': p,
            'wss': wss
        }

    def count_parameters(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total_params, 'trainable': trainable_params}

    def print_architecture(self):
        print("="*80)
        if self.activation_name in ('swish','swish_learnable','lswish') and hasattr(self.activation, 'beta'):
            beta_val = float(self.activation.beta.detach().cpu().item())
            print(f"Architecture: ResNet-PINN | Activation: {self.activation_name.upper()} (β={beta_val:.4f}, learnable)")
        else:
            print(f"Architecture: ResNet-PINN | Activation: {self.activation_name.upper()}")
        print(f"Width: {self.width} | Residual Blocks: {self.blocks}")
        print("="*80)

        counts = self.count_parameters()
        print(f"Total Parameters: {counts['total']:,} (all trainable)")
        print(f"Outputs: [u, v, w, p, wss] -> Velocity (m/s), Pressure (Pa), WSS (Pa)")
        print("="*80)
