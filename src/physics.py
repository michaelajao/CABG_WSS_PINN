"""
Physics Module for Navier-Stokes-based PINN Training

This module implements the physics constraints that make the neural network
"physics-informed". The key equations are:

1. Incompressible Navier-Stokes (momentum conservation):
   ρ(u·∇u) = -∇p + μ∇²u
   
2. Continuity (mass conservation for incompressible flow):
   ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

3. Wall Shear Stress physics constraint:
   Enforces consistency between predicted WSS and velocity gradients

IMPORTANT - Coordinate Scaling:
    Since inputs are normalized to [0, 1] using MinMaxScaler, gradients 
    require chain rule correction to convert from normalized to physical space:
    
    ∂u/∂x_physical = ∂u/∂x_scaled × (1/σ_x)
    
    where σ_x = (x_max - x_min) is the coordinate range.
    This is handled by the coord_scale parameter in all functions.
"""

import torch
import torch.nn as nn
from typing import Tuple
from src.config import RHO, MU


def compute_gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of outputs with respect to inputs using autograd.
    
    Args:
        outputs: Tensor of shape (N, 1) - network outputs
        inputs: Tensor of shape (N, 3) - coordinates with requires_grad=True
        
    Returns:
        Gradient tensor of shape (N, 3) - [∂out/∂x, ∂out/∂y, ∂out/∂z]
    """
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]


def navier_stokes_residual(model: nn.Module, coords: torch.Tensor, 
                           coord_scale: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Navier-Stokes momentum equation residuals.
    
    The incompressible Navier-Stokes equations in steady-state:
        ρ(u·∇u) + ∇p - μ∇²u = 0  (should equal zero if physics satisfied)
    
    Args:
        model: PINN model that outputs {'u', 'v', 'w', 'p'}
        coords: Scaled coordinates (N, 3) - will enable gradients
        coord_scale: Scale factors (1, 3) for chain rule correction
        
    Returns:
        Tuple of (f_u, f_v, f_w) residuals, each shape (N, 1)
        Values close to zero indicate physics is being satisfied.
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']
    
    # First derivatives
    u_g = compute_gradients(u, coords) * coord_scale
    v_g = compute_gradients(v, coords) * coord_scale
    w_g = compute_gradients(w, coords) * coord_scale
    p_g = compute_gradients(p, coords) * coord_scale
    
    # Second derivatives
    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]
    
    # NS residuals: ρ(u·∇u) + ∇p - μ∇²u = 0
    f_u = RHO * (u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]) + \
          p_g[:, 0:1] - MU * (u_xx + u_yy + u_zz)
    f_v = RHO * (u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3]) + \
          p_g[:, 1:2] - MU * (v_xx + v_yy + v_zz)
    f_w = RHO * (u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3]) + \
          p_g[:, 2:3] - MU * (w_xx + w_yy + w_zz)
    
    return f_u, f_v, f_w


def continuity_residual(model: nn.Module, coords: torch.Tensor, 
                        coord_scale: torch.Tensor) -> torch.Tensor:
    """
    Compute continuity equation residual (divergence of velocity).
    
    For incompressible flow: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    
    Args:
        model: PINN model that outputs {'u', 'v', 'w'}
        coords: Scaled coordinates (N, 3)
        coord_scale: Scale factors for chain rule correction
        
    Returns:
        Divergence residual (N, 1). Values close to zero indicate 
        mass conservation is satisfied.
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    
    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale
    
    return u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]


def wss_physics_residual(model: nn.Module, coords: torch.Tensor,
                         normals: torch.Tensor, coord_scale: torch.Tensor
                        ) -> torch.Tensor:
    """
    Compute WSS physics constraint residual.
    
    This enforces consistency between the WSS network output and
    the physics-based WSS computed from velocity gradients.
    
    Physical relationship: WSS = μ * |∂u_tangent/∂n|
    
    Args:
        model: PINN model
        coords: Wall coordinates (scaled)
        normals: Wall normal vectors (N, 3)
        coord_scale: Scale factors for chain rule
        
    Returns:
        Residual: predicted_WSS - computed_WSS (N, 1)
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    
    # Velocity gradients
    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale
    
    # Normal derivatives: ∂u/∂n = ∇u · n
    du_dn = (u_g * normals).sum(dim=1, keepdim=True)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)
    
    # WSS magnitude from velocity gradients
    wss_computed = MU * torch.sqrt(du_dn**2 + dv_dn**2 + dw_dn**2 + 1e-10)
    
    # Predicted WSS from network
    wss_predicted = out['wss']
    
    return wss_predicted - wss_computed
