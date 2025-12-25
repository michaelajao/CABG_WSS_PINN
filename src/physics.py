"""
Physics Module for Navier-Stokes-based PINN Training.

This module implements the physics constraints that make the neural network
"physics-informed". The key equations are:

1. Incompressible Navier-Stokes (momentum conservation):
   rho(u.grad(u)) = -grad(p) + mu * laplacian(u)

2. Continuity (mass conservation for incompressible flow):
   div(u) = du/dx + dv/dy + dw/dz = 0

3. Wall Shear Stress physics constraint:
   Enforces consistency between predicted WSS and velocity gradients.

Note:
    Coordinate Scaling: Since inputs are normalized to [0, 1] using
    MinMaxScaler, gradients require chain rule correction to convert
    from normalized to physical space:

        du/dx_physical = du/dx_scaled * (1/sigma_x)

    where sigma_x = (x_max - x_min) is the coordinate range.
    This is handled by the coord_scale parameter in all functions.

Attributes:
    EPSILON (float): Small constant for numerical stability (1e-10).

Functions:
    compute_gradients: Compute spatial gradients using autograd.
    navier_stokes_residual: Compute N-S momentum equation residuals.
    continuity_residual: Compute divergence of velocity field.
    wss_physics_residual: Compute WSS physics constraint residual.
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.config import MU, RHO

# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON: float = 1e-10  # Small constant for numerical stability


# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of outputs with respect to inputs using autograd.

    Args:
        outputs: Tensor of shape (N, 1) containing network outputs.
        inputs: Tensor of shape (N, 3) containing coordinates with
            requires_grad=True.

    Returns:
        Gradient tensor of shape (N, 3) containing [du/dx, du/dy, du/dz].

    Note:
        This function creates and retains computation graphs for
        higher-order derivatives needed in Navier-Stokes equations.
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


# =============================================================================
# PHYSICS RESIDUALS
# =============================================================================

def navier_stokes_residual(
    model: nn.Module,
    coords: torch.Tensor,
    coord_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Navier-Stokes momentum equation residuals.

    The incompressible Navier-Stokes equations in steady-state:
        rho(u.grad(u)) + grad(p) - mu * laplacian(u) = 0

    Values close to zero indicate physics is being satisfied.

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w', 'p'}.
        coords: Scaled coordinates with shape (N, 3). Gradients will be
            enabled internally.
        coord_scale: Scale factors with shape (1, 3) for chain rule
            correction from normalized to physical coordinates.

    Returns:
        Tuple containing (f_u, f_v, f_w) residuals, each with shape (N, 1).
        These represent the residual of the momentum equation in each
        spatial direction.

    Example:
        >>> coords = torch.randn(100, 3)
        >>> scale = torch.tensor([[0.01, 0.01, 0.01]])  # 10mm range
        >>> f_u, f_v, f_w = navier_stokes_residual(model, coords, scale)
        >>> ns_loss = (f_u**2 + f_v**2 + f_w**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']

    # First derivatives (velocity gradients)
    u_g = compute_gradients(u, coords) * coord_scale
    v_g = compute_gradients(v, coords) * coord_scale
    w_g = compute_gradients(w, coords) * coord_scale
    p_g = compute_gradients(p, coords) * coord_scale

    # Second derivatives (velocity Laplacian components)
    # Apply chain rule twice for second derivatives
    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]

    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]

    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1] * coord_scale[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2] * coord_scale[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3] * coord_scale[:, 2:3]

    # NS residuals: rho(u.grad(u)) + grad(p) - mu * laplacian(u) = 0
    f_u = (
        RHO * (u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3])
        + p_g[:, 0:1]
        - MU * (u_xx + u_yy + u_zz)
    )
    f_v = (
        RHO * (u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3])
        + p_g[:, 1:2]
        - MU * (v_xx + v_yy + v_zz)
    )
    f_w = (
        RHO * (u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3])
        + p_g[:, 2:3]
        - MU * (w_xx + w_yy + w_zz)
    )

    return f_u, f_v, f_w


def continuity_residual(
    model: nn.Module,
    coords: torch.Tensor,
    coord_scale: torch.Tensor
) -> torch.Tensor:
    """
    Compute continuity equation residual (divergence of velocity).

    For incompressible flow: div(u) = du/dx + dv/dy + dw/dz = 0

    Values close to zero indicate mass conservation is satisfied.

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w'}.
        coords: Scaled coordinates with shape (N, 3).
        coord_scale: Scale factors with shape (1, 3) for chain rule
            correction.

    Returns:
        Divergence residual with shape (N, 1).

    Example:
        >>> coords = torch.randn(100, 3)
        >>> scale = torch.tensor([[0.01, 0.01, 0.01]])
        >>> div_u = continuity_residual(model, coords, scale)
        >>> cont_loss = (div_u**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale

    # Divergence: du/dx + dv/dy + dw/dz
    return u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]


def wss_physics_residual(
    model: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor,
    coord_scale: torch.Tensor
) -> torch.Tensor:
    """
    Compute WSS physics constraint residual.

    This enforces consistency between the WSS network output and
    the physics-based WSS computed from velocity gradients at the wall.

    Physical relationship: WSS = mu * |du_tangent/dn|

    Args:
        model: PINN model that outputs dict including 'wss'.
        coords: Wall coordinates (scaled) with shape (N, 3).
        normals: Wall normal vectors with shape (N, 3), should be
            unit vectors pointing outward from the wall.
        coord_scale: Scale factors with shape (1, 3) for chain rule
            correction.

    Returns:
        Residual tensor with shape (N, 1) representing:
        predicted_WSS - computed_WSS. Values close to zero indicate
        the predicted WSS is consistent with the velocity field.

    Note:
        The computed WSS uses the normal derivative of velocity:
        WSS = mu * sqrt((du/dn)^2 + (dv/dn)^2 + (dw/dn)^2)
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # Velocity gradients
    u_g = compute_gradients(out['u'], coords) * coord_scale
    v_g = compute_gradients(out['v'], coords) * coord_scale
    w_g = compute_gradients(out['w'], coords) * coord_scale

    # Normal derivatives: du/dn = grad(u) . n
    du_dn = (u_g * normals).sum(dim=1, keepdim=True)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)

    # WSS magnitude from velocity gradients
    wss_computed = MU * torch.sqrt(
        du_dn**2 + dv_dn**2 + dw_dn**2 + EPSILON
    )

    # Predicted WSS from network
    wss_predicted = out['wss']

    return wss_predicted - wss_computed
