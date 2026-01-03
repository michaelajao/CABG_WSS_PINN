"""
Physics Module for Navier-Stokes-based PINN Training.

This module implements the physics constraints that make the neural network
"physics-informed" using FULLY NON-DIMENSIONAL equations.

NON-DIMENSIONAL FORMULATION:
============================
All quantities are non-dimensionalized using reference scales:
    x* = (x - offset) / L_ref   (coordinates, UNIFORM scaling)
    u* = u / U_ref              (velocity)
    p* = p / (rho * U_ref^2)    (pressure)
    tau* = tau / (mu * U_ref / L_ref) (WSS)

The non-dimensional Navier-Stokes equations become:
    (u*.grad*)u* = -grad*(p*) + (1/Re) * laplacian*(u*)

where Re = rho * U_ref * L_ref / mu is the Reynolds number.

The continuity equation remains:
    div*(u*) = 0

UNIFORM SCALING ADVANTAGE:
=========================
With uniform scaling (same L_ref for all dimensions):
    - Geometry aspect ratios are preserved
    - Gradients are directly in non-dimensional units (no chain rule correction)
    - d/dx* = d/dx_input (since they differ only by constant offset)

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

    With UNIFORM coordinate scaling (same L_ref for all dimensions):
        - Gradients are directly in non-dimensional units
        - No chain rule correction needed

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w', 'p'}.
            All outputs should be in non-dimensional form.
        coords: Non-dimensional coordinates with shape (N, 3).
        Re: Reynolds number (rho * U_ref * L_ref / mu).

    Returns:
        Tuple containing (f_u, f_v, f_w) non-dimensional residuals,
        each with shape (N, 1). Values close to zero indicate the
        momentum equation is satisfied.

    Example:
        >>> coords = torch.randn(100, 3)  # Non-dimensional coordinates
        >>> f_u, f_v, f_w = compute_navier_stokes_residual(model, coords, Re=1500)
        >>> ns_loss = (f_u**2 + f_v**2 + f_w**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # Model outputs are already non-dimensional:
    # u* = u / U_ref, p* = p / (rho * U_ref^2)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']

    # With uniform scaling, gradients are directly in non-dimensional units
    # d/dx* = d/dx_input (no chain rule correction needed)
    u_g = compute_gradients(u, coords)
    v_g = compute_gradients(v, coords)
    w_g = compute_gradients(w, coords)
    p_g = compute_gradients(p, coords)

    # Second derivatives (Laplacian components)
    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3]

    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3]

    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3]

    # Non-dimensional NS residuals:
    # (u*.grad*)u* + grad*(p*) - (1/Re) * laplacian*(u*) = 0
    inv_Re = 1.0 / Re

    f_u = (
        u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]  # Convection
        + p_g[:, 0:1]                                         # Pressure gradient
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
    coords: torch.Tensor
) -> torch.Tensor:
    """
    Compute NON-DIMENSIONAL continuity equation residual.

    For incompressible flow: div*(u*) = du*/dx* + dv*/dy* + dw*/dz* = 0

    With UNIFORM coordinate scaling, gradients are directly in non-dimensional
    units (no chain rule correction needed).

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w'}.
        coords: Non-dimensional coordinates with shape (N, 3).

    Returns:
        Non-dimensional divergence residual with shape (N, 1).

    Example:
        >>> coords = torch.randn(100, 3)  # Non-dimensional coordinates
        >>> div_u = compute_continuity_residual(model, coords)
        >>> cont_loss = (div_u**2).mean()
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # With uniform scaling, gradients are directly in non-dimensional units
    u_g = compute_gradients(out['u'], coords)
    v_g = compute_gradients(out['v'], coords)
    w_g = compute_gradients(out['w'], coords)

    # Non-dimensional divergence: du*/dx* + dv*/dy* + dw*/dz*
    return u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]


def derive_wss_from_velocity_gradients(
    model: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor
) -> torch.Tensor:
    """
    Compute WSS from velocity gradients at wall points.

    This computes the wall shear stress using the tangential component of the
    velocity gradient in the normal direction.

    Physical relationship for Newtonian fluid:
        tau_wall = mu * |du_tangential/dn|

    In non-dimensional form with physics-based WSS scaling (T_ref = mu*U_ref/L_ref):
        tau* = |du*/dn*|_tangential

    Steps:
        1. Compute velocity gradient in normal direction: V_n = (grad(u)·n, ...)
        2. Extract tangential component: V_t = V_n - (V_n · n) * n
        3. WSS* = |V_t|

    With UNIFORM scaling, gradients are directly in non-dimensional units.

    Args:
        model: PINN model that outputs dict with keys {'u', 'v', 'w'}.
        coords: Wall coordinates (non-dimensional) with shape (N, 3).
        normals: Wall normal vectors with shape (N, 3), unit vectors.

    Returns:
        Computed non-dimensional WSS tensor with shape (N, 1).
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # With uniform scaling, gradients are directly in non-dimensional units
    u_g = compute_gradients(out['u'], coords)
    v_g = compute_gradients(out['v'], coords)
    w_g = compute_gradients(out['w'], coords)

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

    # WSS magnitude (non-dimensional): |V_t| = |du*/dn*|_tangential
    wss_nondim = torch.sqrt(
        (vel_grad_tangent ** 2).sum(dim=1, keepdim=True) + EPSILON
    )

    return wss_nondim


def compute_wss_physics_residual(
    model: nn.Module,
    coords: torch.Tensor,
    normals: torch.Tensor,
    scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Compute NON-DIMENSIONAL WSS physics constraint residual.

    This enforces consistency between the predicted WSS and the physics-based
    WSS computed from velocity gradients at the wall.

    Physical relationship for Newtonian fluid:
        tau_wall = mu * |du_tangential/dn|

    In non-dimensional form:
        tau* (physics) = |du*/dn*|_tangential  (scaled by T_ref_physics = mu*U_ref/L_ref)
        tau* (network) = tau / T_ref           (scaled by data-driven T_ref)

    The scale_factor = T_ref_physics / T_ref bridges the two scales.

    Args:
        model: PINN model that outputs dict including 'wss'.
        coords: Wall coordinates (non-dimensional) with shape (N, 3).
        normals: Wall normal vectors with shape (N, 3), unit vectors.
        scale_factor: T_ref_physics / T_ref to convert physics WSS to network scale.

    Returns:
        Non-dimensional residual tensor with shape (N, 1).
        (wss_predicted - scale_factor * wss_physics) normalized for O(1) comparison.
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    # With uniform scaling, gradients are directly in non-dimensional units
    u_g = compute_gradients(out['u'], coords)
    v_g = compute_gradients(out['v'], coords)
    w_g = compute_gradients(out['w'], coords)

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

    # WSS magnitude from physics: |V_t| = |du*/dn*|_tangential
    # This is in physics scale (T_ref_physics = mu*U_ref/L_ref)
    wss_physics = torch.sqrt(
        (vel_grad_tangent ** 2).sum(dim=1, keepdim=True) + EPSILON
    )

    # Predicted WSS from network (in data scale: tau/T_ref)
    wss_predicted = out['wss']

    # Convert physics WSS to network scale and compare
    # scale_factor = T_ref_physics / T_ref (typically small, e.g., 0.001-0.01)
    wss_physics_scaled = wss_physics * scale_factor

    # Normalized residual for stable training
    return wss_predicted - wss_physics_scaled
