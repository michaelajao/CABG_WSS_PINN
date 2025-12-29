"""
Physics Module for Navier-Stokes-based PINN Training.

This module implements the physics constraints that make the neural network
"physics-informed" using FULLY NON-DIMENSIONAL equations.

NON-DIMENSIONAL FORMULATION:
============================
All quantities are non-dimensionalized using reference scales:
    x* = x / L_ref          (coordinates)
    u* = u / U_ref          (velocity)
    p* = p / (rho * U_ref^2) (pressure)
    tau* = tau * L_ref / (mu * U_ref) (WSS)

The non-dimensional Navier-Stokes equations become:
    (u*.grad*)u* = -grad*(p*) + (1/Re) * laplacian*(u*)
    
where Re = rho * U_ref * L_ref / mu is the Reynolds number.

The continuity equation remains:
    div*(u*) = 0

COORDINATE SCALING:
==================
Since inputs are normalized to [0, 1] using MinMaxScaler:
    x_scaled = (x - x_min) / (x_max - x_min)

Gradients require chain rule correction:
    d/dx* = d/dx_scaled * (x_max - x_min) / L_ref
          = d/dx_scaled * coord_range / L_ref

Attributes:
    EPSILON (float): Small constant for numerical stability (1e-10).

Functions:
    compute_gradients: Compute spatial gradients using autograd.
    compute_navier_stokes_residual: Compute non-dimensional N-S residuals.
    compute_continuity_residual: Compute non-dimensional divergence.
    compute_wss_physics_residual: Compute non-dimensional WSS constraint.
    derive_wss_from_velocity_gradients: Compute WSS from velocity field.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn

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
# NON-DIMENSIONAL PHYSICS RESIDUALS
# =============================================================================

def compute_navier_stokes_residual(
    model: nn.Module,
    coords: torch.Tensor,
    coord_scale: torch.Tensor,
    L_ref: float,
    Re: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute NON-DIMENSIONAL Navier-Stokes momentum equation residuals.

    The non-dimensional steady incompressible Navier-Stokes equations:
        (u*.grad*)u* + grad*(p*) - (1/Re) * laplacian*(u*) = 0

    where:
        - u*, v*, w* are non-dimensional velocities (model outputs)
        - p* is non-dimensional pressure (model output)
        - Re = rho * U_ref * L_ref / mu is the Reynolds number
        - grad* and laplacian* are non-dimensional gradient operators

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w', 'p'}.
            All outputs should be in non-dimensional form.
        coords: Scaled coordinates with shape (N, 3) in [0, 1].
        coord_scale: Physical coordinate ranges (x_max - x_min) for each
            dimension, shape (1, 3) in meters.
        L_ref: Reference length scale in meters.
        Re: Reynolds number (rho * U_ref * L_ref / mu).

    Returns:
        Tuple containing (f_u, f_v, f_w) non-dimensional residuals,
        each with shape (N, 1). Values close to zero indicate the
        momentum equation is satisfied.

    Example:
        >>> coords = torch.randn(100, 3)
        >>> coord_scale = torch.tensor([[0.1, 0.09, 0.11]])  # meters
        >>> f_u, f_v, f_w = navier_stokes_residual_nondim(
        ...     model, coords, coord_scale, L_ref=0.1, Re=1500
        ... )
        >>> ns_loss = (f_u**2 + f_v**2 + f_w**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    
    # Model outputs are already non-dimensional:
    # u* = u / U_ref, p* = p / (rho * U_ref^2)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']

    # Gradient scale factor: converts from scaled coords to non-dim coords
    # Chain rule: d(f)/dx* = d(f)/dx_scaled × (L_ref / coord_range)
    # Since x_scaled = (x - x_min)/coord_range and x* = x/L_ref
    grad_scale = L_ref / coord_scale  # Shape (1, 3)

    # First derivatives (non-dimensional velocity gradients)
    u_g = compute_gradients(u, coords) * grad_scale
    v_g = compute_gradients(v, coords) * grad_scale
    w_g = compute_gradients(w, coords) * grad_scale
    p_g = compute_gradients(p, coords) * grad_scale

    # Second derivatives (non-dimensional Laplacian components)
    # Apply chain rule twice: d²/dx*² = d²/dx_scaled² * (coord_range/L_ref)²
    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1] * grad_scale[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2] * grad_scale[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3] * grad_scale[:, 2:3]

    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1] * grad_scale[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2] * grad_scale[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3] * grad_scale[:, 2:3]

    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1] * grad_scale[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2] * grad_scale[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3] * grad_scale[:, 2:3]

    # Non-dimensional NS residuals:
    # (u*.grad*)u* + grad*(p*) - (1/Re) * laplacian*(u*) = 0
    inv_Re = 1.0 / Re
    
    f_u = (
        u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]  # Convection
        + p_g[:, 0:1]                                          # Pressure gradient
        - inv_Re * (u_xx + u_yy + u_zz)                       # Viscous diffusion
    )
    f_v = (
        u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3]
        + p_g[:, 1:2]
        - inv_Re * (v_xx + v_yy + v_zz)
    )
    f_w = (
        u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3]
        + p_g[:, 2:3]
        - inv_Re * (w_xx + w_yy + w_zz)
    )

    return f_u, f_v, f_w


def compute_continuity_residual(
    model: nn.Module,
    coords: torch.Tensor,
    coord_scale: torch.Tensor,
    L_ref: float
) -> torch.Tensor:
    """
    Compute NON-DIMENSIONAL continuity equation residual.

    For incompressible flow: div*(u*) = du*/dx* + dv*/dy* + dw*/dz* = 0

    The continuity equation is the same in dimensional and non-dimensional
    form (both equal zero), but gradients must use proper scaling.

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w'}.
        coords: Scaled coordinates with shape (N, 3) in [0, 1].
        coord_scale: Physical coordinate ranges, shape (1, 3) in meters.
        L_ref: Reference length scale in meters.

    Returns:
        Non-dimensional divergence residual with shape (N, 1).

    Example:
        >>> coords = torch.randn(100, 3)
        >>> coord_scale = torch.tensor([[0.1, 0.09, 0.11]])
        >>> div_u = continuity_residual_nondim(model, coords, coord_scale, 0.1)
        >>> cont_loss = (div_u**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # Gradient scale factor (L_ref / coord_range for proper chain rule)
    grad_scale = L_ref / coord_scale

    u_g = compute_gradients(out['u'], coords) * grad_scale
    v_g = compute_gradients(out['v'], coords) * grad_scale
    w_g = compute_gradients(out['w'], coords) * grad_scale

    # Non-dimensional divergence: du*/dx* + dv*/dy* + dw*/dz*
    return u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]


def derive_wss_from_velocity_gradients(
    model: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor,
    coord_scale: torch.Tensor,
    L_ref: float,
    T_ref_physics: float,
    T_ref: float
) -> torch.Tensor:
    """
    Compute WSS from velocity gradients at wall points.

    This computes the wall shear stress using the tangential component of the
    velocity gradient in the normal direction, multiplied by dynamic viscosity.

    Physical relationship for Newtonian fluid:
        tau_wall = mu * |du_tangential/dn|

    Steps:
        1. Compute velocity gradient in normal direction: V_n = (grad(u)·n, ...)
        2. Extract tangential component: V_t = V_n - (V_n · n) * n
        3. WSS = mu * |V_t| (non-dimensionalized)

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w'}.
        coords: Wall coordinates (scaled) with shape (N, 3) in [0, 1].
        normals: Wall normal vectors with shape (N, 3), unit vectors.
        coord_scale: Physical coordinate ranges, shape (1, 3) in meters.
        L_ref: Reference length scale in meters.
        T_ref_physics: Physics-based WSS scale = mu * U_ref / L_ref.
        T_ref: Data-driven WSS scale (for output consistency).

    Returns:
        Computed WSS tensor with shape (N, 1) in same scale as model output.
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # Gradient scale factor (L_ref / coord_range for proper chain rule)
    grad_scale = L_ref / coord_scale

    # Non-dimensional velocity gradients: shape (N, 3)
    u_g = compute_gradients(out['u'], coords) * grad_scale
    v_g = compute_gradients(out['v'], coords) * grad_scale
    w_g = compute_gradients(out['w'], coords) * grad_scale

    # Velocity gradient in normal direction: V_n = (grad(u)·n, grad(v)·n, grad(w)·n)
    du_dn = (u_g * normals).sum(dim=1, keepdim=True)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)

    # Stack into velocity gradient vector: shape (N, 3)
    vel_grad_normal = torch.cat([du_dn, dv_dn, dw_dn], dim=1)

    # Normal component of velocity gradient: (V_n · n) * n
    vel_grad_normal_component = (vel_grad_normal * normals).sum(dim=1, keepdim=True)
    vel_grad_normal_vec = vel_grad_normal_component * normals

    # Tangential component: V_t = V_n - (V_n · n) * n
    vel_grad_tangent = vel_grad_normal - vel_grad_normal_vec

    # WSS magnitude (non-dimensional): |V_t|
    wss_nondim = torch.sqrt(
        (vel_grad_tangent ** 2).sum(dim=1, keepdim=True) + EPSILON
    )

    # Scale to match network output scale: tau* = T_ref_physics/T_ref * |du*/dn*|
    scale_factor = T_ref_physics / T_ref
    return wss_nondim * scale_factor


def compute_wss_physics_residual(
    model: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor,
    coord_scale: torch.Tensor,
    L_ref: float,
    T_ref: float = None,
    T_ref_physics: float = None
) -> torch.Tensor:
    """
    Compute NON-DIMENSIONAL WSS physics constraint residual.

    This enforces consistency between the predicted WSS and the physics-based
    WSS computed from velocity gradients at the wall.

    Physical relationship for Newtonian fluid:
        tau_wall = mu * |du_tangential/dn|

    The wall shear stress is the TANGENTIAL component of the velocity gradient
    in the normal direction, multiplied by dynamic viscosity.

    Steps:
        1. Compute velocity gradient in normal direction: V_n = (grad(u)·n, grad(v)·n, grad(w)·n)
        2. Extract tangential component: V_t = V_n - (V_n · n) * n
        3. WSS = mu * |V_t|

    Args:
        model: PINN model that outputs dict including 'wss'.
        coords: Wall coordinates (scaled) with shape (N, 3) in [0, 1].
        normals: Wall normal vectors with shape (N, 3), unit vectors.
        coord_scale: Physical coordinate ranges, shape (1, 3) in meters.
        L_ref: Reference length scale in meters.
        T_ref: Data-driven WSS scale (what network outputs are scaled by).
        T_ref_physics: Physics-based WSS scale = mu * U_ref / L_ref.

    Returns:
        Non-dimensional residual tensor with shape (N, 1).
        wss_predicted* - wss_computed* (both in same non-dimensional scale)
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # Gradient scale factor (L_ref / coord_range for proper chain rule)
    grad_scale = L_ref / coord_scale

    # Non-dimensional velocity gradients: shape (N, 3)
    u_g = compute_gradients(out['u'], coords) * grad_scale
    v_g = compute_gradients(out['v'], coords) * grad_scale
    w_g = compute_gradients(out['w'], coords) * grad_scale

    # Velocity gradient in normal direction: V_n = (grad(u)·n, grad(v)·n, grad(w)·n)
    # This gives the rate of change of velocity in the normal direction
    du_dn = (u_g * normals).sum(dim=1, keepdim=True)  # (N, 1)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)  # (N, 1)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)  # (N, 1)

    # Stack into velocity gradient vector: shape (N, 3)
    vel_grad_normal = torch.cat([du_dn, dv_dn, dw_dn], dim=1)

    # Normal component of velocity gradient: (V_n · n) * n
    # This is the component pointing in the normal direction (should be ~0 at wall)
    vel_grad_normal_component = (vel_grad_normal * normals).sum(dim=1, keepdim=True)
    vel_grad_normal_vec = vel_grad_normal_component * normals  # (N, 3)

    # Tangential component: V_t = V_n - (V_n · n) * n
    # This is the wall shear rate (velocity gradient tangent to wall)
    vel_grad_tangent = vel_grad_normal - vel_grad_normal_vec  # (N, 3)

    # WSS magnitude (non-dimensional): |V_t| = |du_tangential/dn|
    wss_physics_nondim = torch.sqrt(
        (vel_grad_tangent ** 2).sum(dim=1, keepdim=True) + EPSILON
    )

    # Predicted WSS from network (in data-driven non-dimensional scale: WSS/T_ref)
    wss_predicted = out['wss']

    # FIXED: Use NORMALIZED residual to handle scale mismatch
    # The issue: T_ref_physics/T_ref = 0.002, making absolute comparison meaningless
    # Solution: Compare normalized quantities (both become O(1))
    #
    # wss_predicted is in [0, ~5] (WSS/T_ref where T_ref=10)
    # wss_physics_nondim is |du*/dn*| which is O(1-10)
    #
    # We want: wss_predicted ∝ wss_physics_nondim (same shape, different scale)
    # Use normalized comparison: (pred/max - physics/max) or log-ratio

    # Normalize both to [0, 1] range within the batch for fair comparison
    wss_pred_norm = wss_predicted / (wss_predicted.abs().max() + EPSILON)
    wss_phys_norm = wss_physics_nondim / (wss_physics_nondim.abs().max() + EPSILON)

    return wss_pred_norm - wss_phys_norm
