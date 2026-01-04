"""
Stability Analysis for Morphology-Augmented Dynamics.

This module implements the NOVEL Morphological Stability Theorem,
providing the first theoretical bound on safe morphology adaptation rates.

Theorem (Morphological Stability):
    Consider the extended system with Lyapunov candidate:
        V(x_ext) = ½q̇ᵀM(q,m)q̇ + U(q,m)
        
    The system is stable if the morphology rate satisfies:
        ‖ṁ‖ ≤ λ_min(K_d) / ‖∂M/∂m‖_F
        
    where K_d is the damping gain and λ_min denotes minimum eigenvalue.

Author: H M Shujaat Zaheer
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.linalg import eigvalsh

from morphollm.core.extended_config_space import ExtendedConfigSpace
from morphollm.core.morphology_dynamics import MorphologyDynamics


@dataclass
class StabilityBounds:
    """Results from stability analysis."""
    max_morphology_rate: float  # Maximum safe ‖ṁ‖
    stability_margin: float     # How far from instability
    lyapunov_derivative: float  # V̇ at current state
    is_stable: bool
    details: Dict = None


class StabilityAnalyzer:
    """
    Stability Analyzer for Morphology-Augmented Systems.
    
    This class implements the Morphological Stability Theorem,
    providing the FIRST theoretical guarantees for safe morphology
    adaptation rates in robotic manipulators.
    
    Key Result:
        The maximum safe morphology rate is:
            ṁ_max = λ_min(K_d) / ‖∂M/∂m‖_F
            
        where:
        - K_d is the damping gain matrix
        - λ_min is the minimum eigenvalue
        - ∂M/∂m is the morphology-Jacobian of the inertia matrix
        - ‖·‖_F is the Frobenius norm
    
    Example:
        >>> analyzer = StabilityAnalyzer(dynamics, damping_gain=np.eye(7) * 10)
        >>> bounds = analyzer.compute_stability_bounds(q, m)
        >>> print(f"Max safe morphology rate: {bounds.max_morphology_rate}")
    """
    
    def __init__(
        self,
        dynamics: MorphologyDynamics,
        damping_gain: Optional[np.ndarray] = None,
        potential_function: Optional[callable] = None
    ):
        """
        Initialize the stability analyzer.
        
        Args:
            dynamics: Morphology-augmented dynamics model
            damping_gain: Damping gain matrix K_d (n×n)
            potential_function: Custom potential U(q,m), default is gravity
        """
        self.dynamics = dynamics
        self.n = dynamics.n
        self.p = dynamics.p
        
        # Default damping gain
        self.K_d = damping_gain if damping_gain is not None else np.eye(self.n) * 10.0
        
        # Potential function (default: gravity potential)
        self.potential = potential_function or self._default_potential
        
    def _default_potential(self, q: np.ndarray, m: np.ndarray) -> float:
        """Default gravitational potential energy."""
        masses = self.dynamics.get_link_masses(m)
        lengths = self.dynamics.get_link_lengths(m)
        g = np.linalg.norm(self.dynamics.config.gravity)
        
        U = 0.0
        for i in range(self.n):
            # Height of link center of mass
            h = sum(lengths[k] * np.sin(sum(q[:k+1])) for k in range(i+1))
            U += masses[i] * g * h
            
        return U
    
    def lyapunov_function(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        m: np.ndarray
    ) -> float:
        """
        Compute the Lyapunov function candidate.
        
        V(x_ext) = ½q̇ᵀM(q,m)q̇ + U(q,m)
        
        Args:
            q: Joint configuration
            q_dot: Joint velocities
            m: Morphology parameters
            
        Returns:
            V: Lyapunov function value
        """
        M = self.dynamics.inertia_matrix(q, m)
        kinetic = 0.5 * q_dot @ M @ q_dot
        potential = self.potential(q, m)
        
        return kinetic + potential
    
    def lyapunov_derivative(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        m: np.ndarray,
        m_dot: np.ndarray,
        tau: np.ndarray
    ) -> float:
        """
        Compute time derivative of Lyapunov function.
        
        V̇ = q̇ᵀ(τ - K_d q̇) + ½q̇ᵀ(∂M/∂m)ṁ q̇ + (∂U/∂m)ṁ
        
        For stability, we need V̇ ≤ 0.
        
        Args:
            q: Joint configuration
            q_dot: Joint velocities  
            m: Morphology parameters
            m_dot: Morphology rates
            tau: Applied torques
            
        Returns:
            V_dot: Time derivative of Lyapunov function
        """
        M = self.dynamics.inertia_matrix(q, m)
        
        # Damping term
        damping_term = -q_dot @ self.K_d @ q_dot
        
        # Power term (work done by torques)
        power_term = q_dot @ tau
        
        # Morphology coupling term (∂M/∂m effect)
        dM_dm = self._compute_dM_dm(q, m)
        morph_kinetic_term = 0.5 * sum(
            q_dot @ dM_dm[j] @ q_dot * m_dot[j]
            for j in range(self.p)
        )
        
        # Potential derivative w.r.t. morphology
        dU_dm = self._compute_dU_dm(q, m)
        morph_potential_term = dU_dm @ m_dot
        
        V_dot = power_term + damping_term + morph_kinetic_term + morph_potential_term
        
        return V_dot
    
    def _compute_dM_dm(
        self,
        q: np.ndarray,
        m: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute morphology-Jacobian of inertia matrix.
        
        Returns:
            dM_dm: Array of shape (p, n, n) where dM_dm[j] = ∂M/∂m_j
        """
        dM_dm = np.zeros((self.p, self.n, self.n))
        M = self.dynamics.inertia_matrix(q, m)
        
        for j in range(self.p):
            m_plus = m.copy()
            m_plus[j] += eps
            M_plus = self.dynamics.inertia_matrix(q, m_plus)
            dM_dm[j] = (M_plus - M) / eps
            
        return dM_dm
    
    def _compute_dU_dm(
        self,
        q: np.ndarray,
        m: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute gradient of potential w.r.t. morphology.
        
        Returns:
            dU_dm: Gradient vector (p,)
        """
        dU_dm = np.zeros(self.p)
        U = self.potential(q, m)
        
        for j in range(self.p):
            m_plus = m.copy()
            m_plus[j] += eps
            U_plus = self.potential(q, m_plus)
            dU_dm[j] = (U_plus - U) / eps
            
        return dU_dm
    
    def compute_morphology_rate_bound(
        self,
        q: np.ndarray,
        m: np.ndarray
    ) -> float:
        """
        Compute the maximum safe morphology rate.
        
        MAIN THEOREM IMPLEMENTATION:
            ṁ_max = λ_min(K_d) / ‖∂M/∂m‖_F
        
        Args:
            q: Joint configuration
            m: Morphology parameters
            
        Returns:
            Maximum safe ‖ṁ‖
        """
        # Minimum eigenvalue of damping gain
        lambda_min = np.min(eigvalsh(self.K_d))
        
        # Frobenius norm of morphology-Jacobian of M
        dM_dm = self._compute_dM_dm(q, m)
        dM_dm_frobenius = np.sqrt(sum(
            np.sum(dM_dm[j]**2) for j in range(self.p)
        ))
        
        # Safety bound
        if dM_dm_frobenius < 1e-10:
            return np.inf  # Morphology has negligible effect
        
        m_dot_max = lambda_min / dM_dm_frobenius
        
        return m_dot_max
    
    def compute_stability_bounds(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        m: np.ndarray,
        m_dot: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None
    ) -> StabilityBounds:
        """
        Compute comprehensive stability bounds.
        
        Args:
            q: Joint configuration
            q_dot: Joint velocities
            m: Morphology parameters
            m_dot: Current morphology rates (optional)
            tau: Applied torques (optional)
            
        Returns:
            StabilityBounds: Complete stability analysis
        """
        m_dot = m_dot if m_dot is not None else np.zeros(self.p)
        tau = tau if tau is not None else -self.K_d @ q_dot  # Default PD control
        
        # Maximum safe rate
        m_dot_max = self.compute_morphology_rate_bound(q, m)
        
        # Current rate magnitude
        current_rate = np.linalg.norm(m_dot)
        
        # Stability margin
        if m_dot_max == np.inf:
            stability_margin = np.inf
        else:
            stability_margin = m_dot_max - current_rate
        
        # Lyapunov derivative
        V_dot = self.lyapunov_derivative(q, q_dot, m, m_dot, tau)
        
        # Stability check
        is_stable = (stability_margin > 0) and (V_dot <= 0)
        
        # Additional details
        details = {
            'lambda_min_Kd': np.min(eigvalsh(self.K_d)),
            'dM_dm_frobenius': np.sqrt(sum(
                np.sum(self._compute_dM_dm(q, m)[j]**2) 
                for j in range(self.p)
            )),
            'current_rate': current_rate,
            'lyapunov_value': self.lyapunov_function(q, q_dot, m)
        }
        
        return StabilityBounds(
            max_morphology_rate=m_dot_max,
            stability_margin=stability_margin,
            lyapunov_derivative=V_dot,
            is_stable=is_stable,
            details=details
        )
    
    def verify_trajectory_stability(
        self,
        q_traj: np.ndarray,
        q_dot_traj: np.ndarray,
        m_traj: np.ndarray,
        m_dot_traj: np.ndarray,
        tau_traj: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Verify stability along an entire trajectory.
        
        Args:
            q_traj: Joint trajectory (T, n)
            q_dot_traj: Velocity trajectory (T, n)
            m_traj: Morphology trajectory (T, p)
            m_dot_traj: Morphology rate trajectory (T, p)
            tau_traj: Torque trajectory (T, n)
            
        Returns:
            is_stable: Overall stability
            stability_margins: Margin at each timestep (T,)
        """
        T = q_traj.shape[0]
        stability_margins = np.zeros(T)
        all_stable = True
        
        for t in range(T):
            bounds = self.compute_stability_bounds(
                q_traj[t], q_dot_traj[t],
                m_traj[t], m_dot_traj[t],
                tau_traj[t]
            )
            stability_margins[t] = bounds.stability_margin
            if not bounds.is_stable:
                all_stable = False
                
        return all_stable, stability_margins
    
    def compute_safe_morphology_rate(
        self,
        q: np.ndarray,
        m: np.ndarray,
        desired_m_dot: np.ndarray,
        safety_factor: float = 0.8
    ) -> np.ndarray:
        """
        Project desired morphology rate to safe region.
        
        Args:
            q: Joint configuration
            m: Morphology parameters
            desired_m_dot: Desired morphology rate
            safety_factor: Fraction of max rate to use (default 0.8)
            
        Returns:
            safe_m_dot: Projected safe morphology rate
        """
        m_dot_max = self.compute_morphology_rate_bound(q, m)
        safe_max = safety_factor * m_dot_max
        
        current_norm = np.linalg.norm(desired_m_dot)
        
        if current_norm <= safe_max:
            return desired_m_dot
        else:
            # Scale down to safe region
            return desired_m_dot * (safe_max / current_norm)


class AdaptiveStabilityController:
    """
    Controller that adapts damping to maintain stability.
    
    If the desired morphology rate would violate stability bounds,
    this controller increases damping to restore the stability margin.
    """
    
    def __init__(
        self,
        analyzer: StabilityAnalyzer,
        base_damping: np.ndarray,
        min_margin: float = 0.1
    ):
        """
        Initialize adaptive stability controller.
        
        Args:
            analyzer: StabilityAnalyzer instance
            base_damping: Base damping gain matrix
            min_margin: Minimum required stability margin
        """
        self.analyzer = analyzer
        self.base_damping = base_damping
        self.min_margin = min_margin
        
    def compute_adaptive_damping(
        self,
        q: np.ndarray,
        m: np.ndarray,
        desired_m_dot: np.ndarray
    ) -> np.ndarray:
        """
        Compute damping gain that ensures stability.
        
        Args:
            q: Joint configuration
            m: Morphology parameters
            desired_m_dot: Desired morphology rate
            
        Returns:
            K_d: Adapted damping gain matrix
        """
        # Required minimum eigenvalue
        dM_dm = self.analyzer._compute_dM_dm(q, m)
        dM_dm_frobenius = np.sqrt(sum(
            np.sum(dM_dm[j]**2) for j in range(self.analyzer.p)
        ))
        
        desired_rate = np.linalg.norm(desired_m_dot)
        required_lambda_min = (desired_rate + self.min_margin) * dM_dm_frobenius
        
        # Current minimum eigenvalue
        current_lambda_min = np.min(eigvalsh(self.base_damping))
        
        if current_lambda_min >= required_lambda_min:
            return self.base_damping
        else:
            # Scale up damping
            scale = required_lambda_min / current_lambda_min
            return self.base_damping * scale
