"""
Morphology-Augmented Dynamics for MorphoLLM.

This module implements the dynamics equations for manipulators with
time-varying morphology, including the morphology-joint coupling term.

The dynamics are governed by:
    M(q,m)q̈ + C(q,q̇,m)q̇ + g(q,m) = τ + J_m^T(q,m)ṁ

where J_m captures the coupling between morphology changes and joint dynamics.

Author: H M Shujaat Zaheer
"""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.linalg import solve

from morphollm.core.extended_config_space import ExtendedConfigSpace


@dataclass
class DynamicsConfig:
    """Configuration for manipulator dynamics."""
    gravity: np.ndarray = None
    friction_coeffs: Optional[np.ndarray] = None
    damping_coeffs: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.gravity is None:
            self.gravity = np.array([0, 0, -9.81])


class MorphologyDynamics:
    """
    Morphology-Augmented Manipulator Dynamics.
    
    Implements the equations of motion for a manipulator with
    time-varying morphology parameters:
    
        M(q,m)q̈ + C(q,q̇,m)q̇ + g(q,m) = τ + J_m^T(q,m)ṁ
        
    where:
        - M(q,m): Inertia matrix (morphology-dependent)
        - C(q,q̇,m): Coriolis/centrifugal matrix
        - g(q,m): Gravity vector
        - τ: Joint torques
        - J_m: Morphology Jacobian (novel contribution)
        - ṁ: Morphology rate of change
    
    The morphology Jacobian J_m captures the coupling between
    morphology changes and joint dynamics - a term absent in all
    existing formulations.
    """
    
    def __init__(
        self,
        config_space: ExtendedConfigSpace,
        config: Optional[DynamicsConfig] = None,
        link_masses: Optional[np.ndarray] = None,
        link_lengths: Optional[np.ndarray] = None
    ):
        """
        Initialize morphology-augmented dynamics.
        
        Args:
            config_space: Extended configuration space
            config: Dynamics configuration
            link_masses: Mass of each link (can be morphology-dependent)
            link_lengths: Length of each link (can be morphology-dependent)
        """
        self.config_space = config_space
        self.config = config or DynamicsConfig()
        self.n = config_space.joint_dim
        self.p = config_space.morphology_dim
        
        # Default link properties
        self.base_link_masses = link_masses if link_masses is not None else np.ones(self.n)
        self.base_link_lengths = link_lengths if link_lengths is not None else np.ones(self.n) * 0.1
        
    def get_link_masses(self, m: np.ndarray) -> np.ndarray:
        """
        Compute morphology-dependent link masses.
        
        Args:
            m: Morphology parameters
            
        Returns:
            Link masses modified by morphology
        """
        # First few morphology params can scale masses
        mass_scale = 1.0 + 0.5 * (m[:min(self.n, self.p)] - 0.5)
        masses = self.base_link_masses.copy()
        masses[:len(mass_scale)] *= mass_scale
        return masses
    
    def get_link_lengths(self, m: np.ndarray) -> np.ndarray:
        """
        Compute morphology-dependent link lengths.
        
        Args:
            m: Morphology parameters
            
        Returns:
            Link lengths modified by morphology
        """
        # Morphology can modify link lengths
        length_scale = 0.8 + 0.4 * m[:min(self.n, self.p)]
        lengths = self.base_link_lengths.copy()
        lengths[:len(length_scale)] *= length_scale
        return lengths
    
    def inertia_matrix(
        self,
        q: np.ndarray,
        m: np.ndarray
    ) -> np.ndarray:
        """
        Compute the morphology-dependent inertia matrix M(q,m).
        
        Args:
            q: Joint configuration (n,)
            m: Morphology parameters (p,)
            
        Returns:
            M: Inertia matrix (n, n)
        """
        masses = self.get_link_masses(m)
        lengths = self.get_link_lengths(m)
        
        # Compute inertia using composite rigid body algorithm
        M = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                # Simplified planar model - extend for 3D
                k = max(i, j)
                for l in range(k, self.n):
                    M[i, j] += masses[l] * lengths[i] * lengths[j]
                    if i <= l and j <= l:
                        angle_diff = sum(q[max(i,j):l+1])
                        M[i, j] *= np.cos(angle_diff) if i != j else 1.0
        
        # Ensure positive definiteness
        M = (M + M.T) / 2 + np.eye(self.n) * 0.01
        
        return M
    
    def coriolis_matrix(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        m: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Coriolis/centrifugal matrix C(q,q̇,m).
        
        Uses Christoffel symbols computed from inertia matrix.
        
        Args:
            q: Joint configuration (n,)
            q_dot: Joint velocities (n,)
            m: Morphology parameters (p,)
            
        Returns:
            C: Coriolis matrix (n, n)
        """
        C = np.zeros((self.n, self.n))
        eps = 1e-6
        
        # Compute Christoffel symbols via numerical differentiation
        M = self.inertia_matrix(q, m)
        
        for k in range(self.n):
            for j in range(self.n):
                for i in range(self.n):
                    # ∂M_kj/∂q_i
                    q_plus = q.copy()
                    q_plus[i] += eps
                    q_minus = q.copy()
                    q_minus[i] -= eps
                    
                    dMkj_dqi = (
                        self.inertia_matrix(q_plus, m)[k, j] -
                        self.inertia_matrix(q_minus, m)[k, j]
                    ) / (2 * eps)
                    
                    # ∂M_ki/∂q_j
                    q_plus = q.copy()
                    q_plus[j] += eps
                    q_minus = q.copy()
                    q_minus[j] -= eps
                    
                    dMki_dqj = (
                        self.inertia_matrix(q_plus, m)[k, i] -
                        self.inertia_matrix(q_minus, m)[k, i]
                    ) / (2 * eps)
                    
                    # ∂M_ij/∂q_k
                    q_plus = q.copy()
                    q_plus[k] += eps
                    q_minus = q.copy()
                    q_minus[k] -= eps
                    
                    dMij_dqk = (
                        self.inertia_matrix(q_plus, m)[i, j] -
                        self.inertia_matrix(q_minus, m)[i, j]
                    ) / (2 * eps)
                    
                    # Christoffel symbol
                    c_ijk = 0.5 * (dMkj_dqi + dMki_dqj - dMij_dqk)
                    C[k, j] += c_ijk * q_dot[i]
        
        return C
    
    def gravity_vector(
        self,
        q: np.ndarray,
        m: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gravity vector g(q,m).
        
        Args:
            q: Joint configuration (n,)
            m: Morphology parameters (p,)
            
        Returns:
            g: Gravity vector (n,)
        """
        masses = self.get_link_masses(m)
        lengths = self.get_link_lengths(m)
        g_vec = self.config.gravity
        
        g = np.zeros(self.n)
        
        for i in range(self.n):
            for j in range(i, self.n):
                # Height of link j center of mass
                h = sum(lengths[k] * np.sin(sum(q[:k+1])) for k in range(j+1))
                g[i] += masses[j] * np.linalg.norm(g_vec) * lengths[i] * \
                        np.cos(sum(q[:i+1]))
        
        return g
    
    def morphology_jacobian(
        self,
        q: np.ndarray,
        m: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Morphology Jacobian J_m(q,m).
        
        This is a NOVEL CONTRIBUTION: captures how morphology changes
        couple to joint dynamics. No existing formulation includes this term.
        
        The morphology Jacobian is defined as:
            J_m[i,j] = ∂(M⁻¹(q,m) * g(q,m))_i / ∂m_j
            
        It represents how changes in morphology parameter j affect
        the acceleration of joint i (in the absence of applied torques).
        
        Args:
            q: Joint configuration (n,)
            m: Morphology parameters (p,)
            
        Returns:
            J_m: Morphology Jacobian (n, p)
        """
        J_m = np.zeros((self.n, self.p))
        eps = 1e-6
        
        # Base values
        M = self.inertia_matrix(q, m)
        g = self.gravity_vector(q, m)
        M_inv_g = solve(M, g)
        
        for j in range(self.p):
            m_plus = m.copy()
            m_plus[j] += eps
            
            M_plus = self.inertia_matrix(q, m_plus)
            g_plus = self.gravity_vector(q, m_plus)
            M_inv_g_plus = solve(M_plus, g_plus)
            
            J_m[:, j] = (M_inv_g_plus - M_inv_g) / eps
        
        return J_m
    
    def forward_dynamics(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        m: np.ndarray,
        m_dot: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint accelerations with morphology coupling.
        
        Full dynamics equation:
            M(q,m)q̈ + C(q,q̇,m)q̇ + g(q,m) = τ + J_m^T(q,m)ṁ
            
        Solving for q̈:
            q̈ = M⁻¹(τ + J_m^T ṁ - Cq̇ - g)
        
        Args:
            q: Joint configuration (n,)
            q_dot: Joint velocities (n,)
            m: Morphology parameters (p,)
            m_dot: Morphology rates (p,)
            tau: Applied joint torques (n,)
            
        Returns:
            q_ddot: Joint accelerations (n,)
        """
        M = self.inertia_matrix(q, m)
        C = self.coriolis_matrix(q, q_dot, m)
        g = self.gravity_vector(q, m)
        J_m = self.morphology_jacobian(q, m)
        
        # Morphology coupling term (novel)
        morphology_coupling = J_m @ m_dot
        
        # Total generalized force
        total_force = tau + morphology_coupling - C @ q_dot - g
        
        # Solve for acceleration
        q_ddot = solve(M, total_force)
        
        return q_ddot
    
    def inverse_dynamics(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ddot: np.ndarray,
        m: np.ndarray,
        m_dot: np.ndarray
    ) -> np.ndarray:
        """
        Compute required torques for desired motion.
        
        τ = M(q,m)q̈ + C(q,q̇,m)q̇ + g(q,m) - J_m^T(q,m)ṁ
        
        Args:
            q: Joint configuration (n,)
            q_dot: Joint velocities (n,)
            q_ddot: Desired joint accelerations (n,)
            m: Morphology parameters (p,)
            m_dot: Morphology rates (p,)
            
        Returns:
            tau: Required joint torques (n,)
        """
        M = self.inertia_matrix(q, m)
        C = self.coriolis_matrix(q, q_dot, m)
        g = self.gravity_vector(q, m)
        J_m = self.morphology_jacobian(q, m)
        
        tau = M @ q_ddot + C @ q_dot + g - J_m @ m_dot
        
        return tau
    
    def step(
        self,
        x_ext: np.ndarray,
        x_ext_dot: np.ndarray,
        tau: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one timestep of the dynamics.
        
        Args:
            x_ext: Extended state [q, m]
            x_ext_dot: Extended velocity [q̇, ṁ]
            tau: Applied torques
            dt: Timestep
            
        Returns:
            x_ext_new: New extended state
            x_ext_dot_new: New extended velocity
        """
        q, m = self.config_space.decompose(x_ext)
        q_dot, m_dot = self.config_space.decompose_control(x_ext_dot)
        
        # Compute acceleration
        q_ddot = self.forward_dynamics(q, q_dot, m, m_dot, tau)
        
        # Semi-implicit Euler integration
        q_dot_new = q_dot + q_ddot * dt
        q_new = q + q_dot_new * dt
        m_new = m + m_dot * dt  # Morphology follows commanded rate
        
        # Clip to bounds
        x_ext_new = self.config_space.compose(q_new, m_new)
        x_ext_new = self.config_space.clip_state(x_ext_new)
        
        x_ext_dot_new = self.config_space.compose_control(q_dot_new, m_dot)
        
        return x_ext_new, x_ext_dot_new


class DifferentiableMorphologyDynamics(nn.Module):
    """
    PyTorch-differentiable version of morphology dynamics.
    
    Enables gradient-based optimization through dynamics.
    """
    
    def __init__(self, config_space: ExtendedConfigSpace):
        super().__init__()
        self.n = config_space.joint_dim
        self.p = config_space.morphology_dim
        
        # Learnable dynamics parameters
        self.base_masses = nn.Parameter(torch.ones(self.n))
        self.base_lengths = nn.Parameter(torch.ones(self.n) * 0.1)
        
    def forward(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
        m: torch.Tensor,
        m_dot: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint accelerations (differentiable).
        
        Args:
            q: Joint configuration (batch, n)
            q_dot: Joint velocities (batch, n)
            m: Morphology parameters (batch, p)
            m_dot: Morphology rates (batch, p)
            tau: Applied torques (batch, n)
            
        Returns:
            q_ddot: Joint accelerations (batch, n)
        """
        batch_size = q.shape[0]
        
        # Morphology-dependent masses
        mass_scale = 1.0 + 0.5 * (m[:, :self.n] - 0.5)
        masses = self.base_masses.unsqueeze(0) * mass_scale
        
        # Simplified diagonal inertia (extend for full model)
        M_diag = masses * self.base_lengths.unsqueeze(0) ** 2
        M_inv = 1.0 / (M_diag + 1e-6)
        
        # Simplified gravity
        g = masses * 9.81 * self.base_lengths.unsqueeze(0) * torch.cos(q)
        
        # Morphology coupling (simplified)
        J_m = torch.zeros(batch_size, self.n, self.p, device=q.device)
        J_m[:, :, :self.n] = torch.diag_embed(
            0.1 * self.base_lengths.unsqueeze(0).expand(batch_size, -1)
        )
        morphology_force = torch.einsum('bnp,bp->bn', J_m, m_dot)
        
        # Acceleration
        q_ddot = M_inv * (tau + morphology_force - g)
        
        return q_ddot
