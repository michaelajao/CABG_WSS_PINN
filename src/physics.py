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
# CARREAU-YASUDA RHEOLOGY
# =============================================================================

def carreau_yasuda_viscosity(
    gamma_dot: torch.Tensor,
    mu_inf: float,
    mu_0: float,
    lam: float,
    n: float,
    a: float,
) -> torch.Tensor:
    """Pointwise Carreau-Yasuda effective viscosity.

    mu_eff = mu_inf + (mu_0 - mu_inf) * [1 + (lam * gamma_dot)^a]^((n-1)/a)

    Args:
        gamma_dot: Shear-rate magnitude [1/s], shape (N, 1).
        mu_inf, mu_0, lam, n, a: Carreau-Yasuda parameters (SI units).
    Returns:
        Effective viscosity tensor with shape (N, 1).
    """
    base = 1.0 + (lam * (gamma_dot + EPSILON)).pow(a)
    return mu_inf + (mu_0 - mu_inf) * base.pow((n - 1.0) / a)


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
    Re: float,
    rheology: str = "newtonian",
    cy_params: Dict[str, float] = None,
    U_ref: float = None,
    L_ref: float = None,
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
    # (u*.grad*)u* + grad*(p*) - (1/Re_local) * laplacian*(u*) = 0
    # For Newtonian flow Re_local = Re (constant). For Carreau-Yasuda, the
    # effective viscosity varies pointwise with the local shear rate, so
    # 1/Re_local = (mu_eff(gamma_dot) / mu_inf) * (1/Re).
    inv_Re_base = 1.0 / Re

    if rheology == "carreau_yasuda":
        if cy_params is None or U_ref is None or L_ref is None:
            raise ValueError(
                "carreau_yasuda rheology requires cy_params, U_ref, and L_ref"
            )
        # Symmetric strain-rate tensor in non-dim units, then convert to s^-1
        # via gamma_dot = gamma_dot* * (U_ref / L_ref) since coordinates are
        # uniformly scaled by L_ref and velocities by U_ref.
        D11 = u_g[:, 0:1]
        D22 = v_g[:, 1:2]
        D33 = w_g[:, 2:3]
        D12 = 0.5 * (u_g[:, 1:2] + v_g[:, 0:1])
        D13 = 0.5 * (u_g[:, 2:3] + w_g[:, 0:1])
        D23 = 0.5 * (v_g[:, 2:3] + w_g[:, 1:2])
        DD = (D11.pow(2) + D22.pow(2) + D33.pow(2)
              + 2.0 * (D12.pow(2) + D13.pow(2) + D23.pow(2)))
        gamma_dot_nondim = torch.sqrt(2.0 * DD.clamp(min=0.0) + EPSILON)
        gamma_dot = gamma_dot_nondim * (U_ref / L_ref)
        mu_eff = carreau_yasuda_viscosity(gamma_dot, **cy_params)
        # mu_ratio = mu_eff / mu_inf -> inv_Re_local = mu_ratio * inv_Re
        mu_ratio = mu_eff / cy_params["mu_inf"]
        inv_Re_local = mu_ratio * inv_Re_base
    else:
        inv_Re_local = inv_Re_base

    f_u = (
        u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]  # Convection
        + p_g[:, 0:1]                                         # Pressure gradient
        - inv_Re_local * (u_xx + u_yy + u_zz)                 # Viscous diffusion
    )
    f_v = (
        u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3]
        + p_g[:, 1:2]
        - inv_Re_local * (v_xx + v_yy + v_zz)
    )
    f_w = (
        u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3]
        + p_g[:, 2:3]
        - inv_Re_local * (w_xx + w_yy + w_zz)
    )

    return f_u, f_v, f_w


def compute_physics_residuals_fused(
    model: nn.Module,
    coords: torch.Tensor,
    Re: float,
    rheology: str = "newtonian",
    cy_params: Dict[str, float] = None,
    U_ref: float = None,
    L_ref: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pass fused N-S momentum + continuity residuals on collocation points.

    Equivalent to running ``compute_navier_stokes_residual`` and
    ``compute_continuity_residual`` back-to-back on the same coords, but with
    one forward pass through the model and one set of first derivatives shared
    by both residuals. Roughly halves the host-side autograd-graph cost per
    iteration for the physics loss.

    Returns:
        Tuple (f_u, f_v, f_w, div_u) of non-dimensional residuals, each (N, 1).
    """
    coords = coords.requires_grad_(True)
    out = model(coords)
    u, v, w, p = out['u'], out['v'], out['w'], out['p']

    u_g = compute_gradients(u, coords)
    v_g = compute_gradients(v, coords)
    w_g = compute_gradients(w, coords)
    p_g = compute_gradients(p, coords)

    u_xx = compute_gradients(u_g[:, 0:1], coords)[:, 0:1]
    u_yy = compute_gradients(u_g[:, 1:2], coords)[:, 1:2]
    u_zz = compute_gradients(u_g[:, 2:3], coords)[:, 2:3]

    v_xx = compute_gradients(v_g[:, 0:1], coords)[:, 0:1]
    v_yy = compute_gradients(v_g[:, 1:2], coords)[:, 1:2]
    v_zz = compute_gradients(v_g[:, 2:3], coords)[:, 2:3]

    w_xx = compute_gradients(w_g[:, 0:1], coords)[:, 0:1]
    w_yy = compute_gradients(w_g[:, 1:2], coords)[:, 1:2]
    w_zz = compute_gradients(w_g[:, 2:3], coords)[:, 2:3]

    inv_Re_base = 1.0 / Re
    if rheology == "carreau_yasuda":
        if cy_params is None or U_ref is None or L_ref is None:
            raise ValueError(
                "carreau_yasuda rheology requires cy_params, U_ref, and L_ref"
            )
        D11 = u_g[:, 0:1]
        D22 = v_g[:, 1:2]
        D33 = w_g[:, 2:3]
        D12 = 0.5 * (u_g[:, 1:2] + v_g[:, 0:1])
        D13 = 0.5 * (u_g[:, 2:3] + w_g[:, 0:1])
        D23 = 0.5 * (v_g[:, 2:3] + w_g[:, 1:2])
        DD = (D11.pow(2) + D22.pow(2) + D33.pow(2)
              + 2.0 * (D12.pow(2) + D13.pow(2) + D23.pow(2)))
        gamma_dot_nondim = torch.sqrt(2.0 * DD.clamp(min=0.0) + EPSILON)
        gamma_dot = gamma_dot_nondim * (U_ref / L_ref)
        mu_eff = carreau_yasuda_viscosity(gamma_dot, **cy_params)
        inv_Re_local = (mu_eff / cy_params["mu_inf"]) * inv_Re_base
    else:
        inv_Re_local = inv_Re_base

    f_u = (u * u_g[:, 0:1] + v * u_g[:, 1:2] + w * u_g[:, 2:3]
           + p_g[:, 0:1] - inv_Re_local * (u_xx + u_yy + u_zz))
    f_v = (u * v_g[:, 0:1] + v * v_g[:, 1:2] + w * v_g[:, 2:3]
           + p_g[:, 1:2] - inv_Re_local * (v_xx + v_yy + v_zz))
    f_w = (u * w_g[:, 0:1] + v * w_g[:, 1:2] + w * w_g[:, 2:3]
           + p_g[:, 2:3] - inv_Re_local * (w_xx + w_yy + w_zz))

    div_u = u_g[:, 0:1] + v_g[:, 1:2] + w_g[:, 2:3]
    return f_u, f_v, f_w, div_u


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
    scale_factor: float = 1.0,
    rheology: str = "newtonian",
    cy_params: Dict[str, float] = None,
    U_ref: float = None,
    L_ref: float = None,
) -> torch.Tensor:
    """Compute NON-DIMENSIONAL WSS physics constraint residual.

    Physical relationship: tau_wall = mu_eff * |du_tangential/dn|
    where mu_eff is constant for Newtonian flow and a pointwise function of
    the local shear rate for Carreau-Yasuda flow.

    The non-dimensional WSS in the PHYSICS scale (T_ref_physics = mu_inf * U_ref / L_ref):
        tau*_phys = (mu_eff / mu_inf) * |du*/dn*|_tangential
    For Newtonian flow this reduces to |du*/dn*|_tangential since mu_eff = mu_inf.
    The bridge to the data-driven NETWORK scale (tau / T_ref) is the
    scale_factor = T_ref_physics / T_ref.

    Under "carreau_yasuda" the residual respects the spatially varying viscosity;
    this is the formulation required when the data-driven CFD targets are
    non-Newtonian. Pairing this rheology flag with Newtonian CFD data is
    physically inconsistent and is rejected by an assertion in main.py.

    Args:
        model: PINN model that outputs dict including 'wss'.
        coords: Wall coordinates (non-dimensional) with shape (N, 3).
        normals: Wall normal vectors with shape (N, 3), unit vectors.
        scale_factor: T_ref_physics / T_ref bridging physics and network scales.
        rheology: "newtonian" (default) or "carreau_yasuda".
        cy_params: Carreau-Yasuda parameters dict with keys
            {mu_inf, mu_0, lam, n, a}. Required for "carreau_yasuda".
        U_ref: Velocity reference scale (m/s). Required for "carreau_yasuda".
        L_ref: Length reference scale (m). Required for "carreau_yasuda".

    Returns:
        Non-dimensional residual tensor with shape (N, 1):
            wss_predicted - scale_factor * (mu_eff/mu_inf) * |du*/dn*|_tangential.
    """
    coords = coords.requires_grad_(True)
    out = model(coords)

    u_g = compute_gradients(out['u'], coords)
    v_g = compute_gradients(out['v'], coords)
    w_g = compute_gradients(out['w'], coords)

    du_dn = (u_g * normals).sum(dim=1, keepdim=True)
    dv_dn = (v_g * normals).sum(dim=1, keepdim=True)
    dw_dn = (w_g * normals).sum(dim=1, keepdim=True)
    vel_grad_normal = torch.cat([du_dn, dv_dn, dw_dn], dim=1)

    vel_grad_normal_component = (vel_grad_normal * normals).sum(dim=1, keepdim=True)
    vel_grad_tangent = vel_grad_normal - vel_grad_normal_component * normals

    # WSS magnitude in the PHYSICS scale (T_ref_physics = mu_inf * U_ref / L_ref).
    wss_physics = torch.sqrt(
        (vel_grad_tangent ** 2).sum(dim=1, keepdim=True) + EPSILON
    )

    if rheology == "carreau_yasuda":
        if cy_params is None or U_ref is None or L_ref is None:
            raise ValueError(
                "carreau_yasuda rheology requires cy_params, U_ref, and L_ref"
            )
        # Symmetric strain-rate tensor in non-dim units.
        D11 = u_g[:, 0:1]
        D22 = v_g[:, 1:2]
        D33 = w_g[:, 2:3]
        D12 = 0.5 * (u_g[:, 1:2] + v_g[:, 0:1])
        D13 = 0.5 * (u_g[:, 2:3] + w_g[:, 0:1])
        D23 = 0.5 * (v_g[:, 2:3] + w_g[:, 1:2])
        DD = (D11.pow(2) + D22.pow(2) + D33.pow(2)
              + 2.0 * (D12.pow(2) + D13.pow(2) + D23.pow(2)))
        gamma_dot_nondim = torch.sqrt(2.0 * DD.clamp(min=0.0) + EPSILON)
        gamma_dot = gamma_dot_nondim * (U_ref / L_ref)
        mu_eff = carreau_yasuda_viscosity(gamma_dot, **cy_params)
        mu_ratio = mu_eff / cy_params["mu_inf"]
        wss_physics = mu_ratio * wss_physics

    wss_predicted = out['wss']
    return wss_predicted - wss_physics * scale_factor
