"""
Physics-informed loss functions

Contains implementations of physics equations used as constraints:
- Navier-Stokes equations (momentum conservation)
- Continuity equation (mass conservation)
"""

import torch
from typing import Tuple, Optional
import numpy as np

from .config import RHO, MU
from .model import compute_gradients


def navier_stokes_residual(
    model: torch.nn.Module,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Navier-Stokes residual for physics loss

    The incompressible Navier-Stokes equations in steady state:
        ρ(u·∇u) = -∇p + μ∇²u + f

    For steady-state flow without body forces:
        ρ(u·∇u) = -∇p + μ∇²u

    Expanded for each component:
        ρ(u ∂u/∂x + v ∂u/∂y + w ∂u/∂z) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
        ρ(u ∂v/∂x + v ∂v/∂y + w ∂v/∂z) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
        ρ(u ∂w/∂x + v ∂w/∂y + w ∂w/∂z) = -∂p/∂z + μ(∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)

    Args:
        model: PINN model
        coords: Spatial coordinates (batch_size, 3) [x, y, z] - SCALED inputs
                Must have requires_grad=True
        coord_scale: Scale factors for coordinate transformation (1, 3)
                     If coords are scaled: ∂/∂x_physical = (1/scale) * ∂/∂x_scaled
                     If None, assumes coords are in physical units (no scaling)

    Returns:
        Tuple of (f_u, f_v, f_w) residuals for each momentum equation
        Each residual should be close to zero if physics is satisfied
    """
    coords.requires_grad_(True)

    # Get coordinate scaling factors (chain rule correction)
    # MinMaxScaler: x_scaled = (x - min) / (max - min)
    # ∂/∂x_physical = (max - min) * ∂/∂x_scaled
    if coord_scale is None:
        coord_scale = torch.ones(1, 3, device=coords.device)

    # Forward pass to get velocity and pressure
    outputs = model(coords)
    u = outputs['u']  # x-component of velocity
    v = outputs['v']  # y-component of velocity
    w = outputs['w']  # z-component of velocity
    p = outputs['pressure']

    # Compute first derivatives of velocity components (w.r.t. scaled coords)
    # Derivatives of u
    u_x_scaled = compute_gradients(u, coords)[:, 0:1]
    u_y_scaled = compute_gradients(u, coords)[:, 1:2]
    u_z_scaled = compute_gradients(u, coords)[:, 2:3]

    # Derivatives of v
    v_x_scaled = compute_gradients(v, coords)[:, 0:1]
    v_y_scaled = compute_gradients(v, coords)[:, 1:2]
    v_z_scaled = compute_gradients(v, coords)[:, 2:3]

    # Derivatives of w
    w_x_scaled = compute_gradients(w, coords)[:, 0:1]
    w_y_scaled = compute_gradients(w, coords)[:, 1:2]
    w_z_scaled = compute_gradients(w, coords)[:, 2:3]

    # Pressure gradients
    p_x_scaled = compute_gradients(p, coords)[:, 0:1]
    p_y_scaled = compute_gradients(p, coords)[:, 1:2]
    p_z_scaled = compute_gradients(p, coords)[:, 2:3]

    # Apply chain rule: ∂/∂x_physical = (max - min) * ∂/∂x_scaled
    # Extract scale factors (data_range for MinMaxScaler)
    scale_x = coord_scale[:, 0:1]
    scale_y = coord_scale[:, 1:2]
    scale_z = coord_scale[:, 2:3]

    # First derivatives in physical coordinates
    u_x = u_x_scaled * scale_x
    u_y = u_y_scaled * scale_y
    u_z = u_z_scaled * scale_z

    v_x = v_x_scaled * scale_x
    v_y = v_y_scaled * scale_y
    v_z = v_z_scaled * scale_z

    w_x = w_x_scaled * scale_x
    w_y = w_y_scaled * scale_y
    w_z = w_z_scaled * scale_z

    p_x = p_x_scaled * scale_x
    p_y = p_y_scaled * scale_y
    p_z = p_z_scaled * scale_z

    # Compute second derivatives (Laplacian components)
    # Second derivatives of u (w.r.t. scaled coords, then apply chain rule)
    u_xx_scaled = compute_gradients(u_x_scaled, coords)[:, 0:1]
    u_yy_scaled = compute_gradients(u_y_scaled, coords)[:, 1:2]
    u_zz_scaled = compute_gradients(u_z_scaled, coords)[:, 2:3]

    # Second derivatives of v
    v_xx_scaled = compute_gradients(v_x_scaled, coords)[:, 0:1]
    v_yy_scaled = compute_gradients(v_y_scaled, coords)[:, 1:2]
    v_zz_scaled = compute_gradients(v_z_scaled, coords)[:, 2:3]

    # Second derivatives of w
    w_xx_scaled = compute_gradients(w_x_scaled, coords)[:, 0:1]
    w_yy_scaled = compute_gradients(w_y_scaled, coords)[:, 1:2]
    w_zz_scaled = compute_gradients(w_z_scaled, coords)[:, 2:3]

    # Apply chain rule for second derivatives: ∂²/∂x² = (max-min)² * ∂²/∂x_scaled²
    u_xx = u_xx_scaled * (scale_x ** 2)
    u_yy = u_yy_scaled * (scale_y ** 2)
    u_zz = u_zz_scaled * (scale_z ** 2)

    v_xx = v_xx_scaled * (scale_x ** 2)
    v_yy = v_yy_scaled * (scale_y ** 2)
    v_zz = v_zz_scaled * (scale_z ** 2)

    w_xx = w_xx_scaled * (scale_x ** 2)
    w_yy = w_yy_scaled * (scale_y ** 2)
    w_zz = w_zz_scaled * (scale_z ** 2)

    # Navier-Stokes residuals
    # f_u: x-momentum equation residual
    convection_u = u * u_x + v * u_y + w * u_z
    diffusion_u = u_xx + u_yy + u_zz
    f_u = RHO * convection_u + p_x - MU * diffusion_u

    # f_v: y-momentum equation residual
    convection_v = u * v_x + v * v_y + w * v_z
    diffusion_v = v_xx + v_yy + v_zz
    f_v = RHO * convection_v + p_y - MU * diffusion_v

    # f_w: z-momentum equation residual
    convection_w = u * w_x + v * w_y + w * w_z
    diffusion_w = w_xx + w_yy + w_zz
    f_w = RHO * convection_w + p_z - MU * diffusion_w

    return f_u, f_v, f_w


def continuity_residual(
    model: torch.nn.Module,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute continuity equation residual

    The continuity equation for incompressible flow:
        ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    This ensures mass conservation in the fluid.

    Args:
        model: PINN model
        coords: Spatial coordinates (batch_size, 3) [x, y, z] - SCALED inputs
                Must have requires_grad=True
        coord_scale: Scale factors for coordinate transformation (1, 3)
                     If coords are scaled: ∂/∂x_physical = (1/scale) * ∂/∂x_scaled
                     If None, assumes coords are in physical units (no scaling)

    Returns:
        Continuity residual (batch_size, 1)
        Should be close to zero if mass is conserved
    """
    coords.requires_grad_(True)

    # Get coordinate scaling factors (chain rule correction)
    # MinMaxScaler: x_scaled = (x - min) / (max - min)
    # ∂/∂x_physical = (max - min) * ∂/∂x_scaled
    if coord_scale is None:
        coord_scale = torch.ones(1, 3, device=coords.device)

    # Forward pass to get velocity components
    outputs = model(coords)
    u = outputs['u']
    v = outputs['v']
    w = outputs['w']

    # Compute velocity divergence (w.r.t. scaled coords)
    u_x_scaled = compute_gradients(u, coords)[:, 0:1]
    v_y_scaled = compute_gradients(v, coords)[:, 1:2]
    w_z_scaled = compute_gradients(w, coords)[:, 2:3]

    # Apply chain rule to get physical gradients
    scale_x = coord_scale[:, 0:1]
    scale_y = coord_scale[:, 1:2]
    scale_z = coord_scale[:, 2:3]

    u_x = u_x_scaled * scale_x
    v_y = v_y_scaled * scale_y
    w_z = w_z_scaled * scale_z

    # Continuity residual
    continuity = u_x + v_y + w_z

    return continuity


def compute_physics_loss(
    model: torch.nn.Module,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
    weight_nse: float = 1.0,
    weight_cont: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined physics loss from Navier-Stokes and continuity

    Args:
        model: PINN model
        coords: Spatial coordinates (batch_size, 3) - SCALED inputs
        coord_scale: Scale factors for coordinate transformation (1, 3)
        weight_nse: Weight for Navier-Stokes loss
        weight_cont: Weight for continuity loss

    Returns:
        Tuple of (nse_loss, continuity_loss, total_physics_loss)
    """
    # Navier-Stokes loss
    f_u, f_v, f_w = navier_stokes_residual(model, coords, coord_scale)
    loss_nse = (f_u**2 + f_v**2 + f_w**2).mean()

    # Continuity loss
    cont = continuity_residual(model, coords, coord_scale)
    loss_cont = (cont**2).mean()

    # Combined physics loss
    loss_physics = weight_nse * loss_nse + weight_cont * loss_cont

    return loss_nse, loss_cont, loss_physics


def wss_physics_residual(
    model: torch.nn.Module,
    coords: torch.Tensor,
    wall_normals: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute WSS physics residual at wall points

    Physical definition of Wall Shear Stress:
        WSS = μ * |∂u/∂n|  where n is wall normal direction

    More precisely:
        WSS = μ * sqrt((∂u/∂n)² + (∂v/∂n)² + (∂w/∂n)²)

    Where ∂u/∂n = ∇u · n (directional derivative along normal)

    This constraint enforces that the model's WSS output matches
    the physics-based calculation from velocity gradients.

    Args:
        model: PINN model
        coords: Wall coordinates (batch_size, 3) - SCALED inputs
        wall_normals: Wall normal vectors (batch_size, 3) - unit vectors
        coord_scale: Scale factors for coordinate transformation (1, 3)

    Returns:
        WSS residual (batch_size, 1): difference between predicted WSS
        and physics-based WSS from velocity gradients
    """
    coords.requires_grad_(True)

    # Get coordinate scaling
    if coord_scale is None:
        coord_scale = torch.ones(1, 3, device=coords.device)

    # Forward pass
    outputs = model(coords)
    u = outputs['u']
    v = outputs['v']
    w = outputs['w']
    wss_pred = outputs['wss']  # Model's predicted WSS

    # Compute velocity gradients (∇u, ∇v, ∇w)
    u_grads_scaled = compute_gradients(u, coords)  # (batch, 3)
    v_grads_scaled = compute_gradients(v, coords)
    w_grads_scaled = compute_gradients(w, coords)

    # Apply chain rule for physical gradients
    scale_x = coord_scale[:, 0:1]
    scale_y = coord_scale[:, 1:2]
    scale_z = coord_scale[:, 2:3]
    coord_scale_vec = torch.cat([scale_x, scale_y, scale_z], dim=1)  # (1, 3)

    u_grads = u_grads_scaled * coord_scale_vec  # (batch, 3)
    v_grads = v_grads_scaled * coord_scale_vec
    w_grads = w_grads_scaled * coord_scale_vec

    # Compute normal derivatives: ∂u/∂n = ∇u · n
    du_dn = (u_grads * wall_normals).sum(dim=1, keepdim=True)  # (batch, 1)
    dv_dn = (v_grads * wall_normals).sum(dim=1, keepdim=True)
    dw_dn = (w_grads * wall_normals).sum(dim=1, keepdim=True)

    # Physics-based WSS: τ_wall = μ * |∂u/∂n|
    wss_physics = MU * torch.sqrt(du_dn**2 + dv_dn**2 + dw_dn**2 + 1e-8)

    # Residual: difference between predicted and physics-based WSS
    wss_residual = wss_pred - wss_physics

    return wss_residual


def check_physics_residuals(
    model: torch.nn.Module,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> dict:
    """
    Check physics residuals for diagnostics

    Args:
        model: PINN model
        coords: Spatial coordinates (batch_size, 3) - SCALED inputs
        coord_scale: Scale factors for coordinate transformation (1, 3)
        verbose: Whether to print results

    Returns:
        Dictionary with residual statistics
    """
    model.eval()

    with torch.no_grad():
        coords_grad = coords.clone().requires_grad_(True)

        # Compute residuals
        f_u, f_v, f_w = navier_stokes_residual(model, coords_grad, coord_scale)
        cont = continuity_residual(model, coords_grad, coord_scale)

        # Compute statistics
        stats = {
            'nse_u_mean': f_u.abs().mean().item(),
            'nse_u_max': f_u.abs().max().item(),
            'nse_v_mean': f_v.abs().mean().item(),
            'nse_v_max': f_v.abs().max().item(),
            'nse_w_mean': f_w.abs().mean().item(),
            'nse_w_max': f_w.abs().max().item(),
            'continuity_mean': cont.abs().mean().item(),
            'continuity_max': cont.abs().max().item()
        }

        if verbose:
            print("\n" + "="*80)
            print("PHYSICS RESIDUALS CHECK")
            print("="*80)
            print("\nNavier-Stokes Residuals:")
            print(f"  f_u (x-momentum): Mean = {stats['nse_u_mean']:.4e}, Max = {stats['nse_u_max']:.4e}")
            print(f"  f_v (y-momentum): Mean = {stats['nse_v_mean']:.4e}, Max = {stats['nse_v_max']:.4e}")
            print(f"  f_w (z-momentum): Mean = {stats['nse_w_mean']:.4e}, Max = {stats['nse_w_max']:.4e}")
            print("\nContinuity Residual:")
            print(f"  ∇·u: Mean = {stats['continuity_mean']:.4e}, Max = {stats['continuity_max']:.4e}")
            print("="*80)

    return stats
