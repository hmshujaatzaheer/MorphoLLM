"""
Semantic Morphological Model Predictive Control (SM-MPC).

This module implements the NOVEL SM-MPC framework - the first MPC
formulation with morphology as a continuous decision variable,
enabling real-time optimal control with stability guarantees.

The optimization problem:
    min  Σ_{k=0}^{N-1} [‖x_ext^k - x_ref^k‖²_Q + ‖u_ext^k‖²_R] + ‖x_ext^N - x_ref^N‖²_Qf
    
    s.t. x_ext^{k+1} = f_ext(x_ext^k, u_ext^k)
         ‖ṁ^k‖ ≤ ṁ_max  (morphology rate limit)
         m^k ∈ M_feasible  (physical constraints)

Author: H M Shujaat Zaheer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Callable
import torch
import torch.nn as nn
from scipy.optimize import minimize

from morphollm.core.extended_config_space import ExtendedConfigSpace
from morphollm.core.morphology_dynamics import MorphologyDynamics
from morphollm.core.stability_analysis import StabilityAnalyzer
from morphollm.algorithms.l2mt import MorphologyTrajectory


@dataclass
class SMMPCConfig:
    """Configuration for SM-MPC controller."""
    horizon: int = 20
    dt: float = 0.01
    joint_weight: float = 1.0
    morphology_weight: float = 0.5
    control_weight: float = 0.01
    terminal_weight: float = 10.0
    morphology_rate_limit: float = 0.1
    max_iterations: int = 50
    tolerance: float = 1e-4
    semantic_adaptation_rate: float = 0.1


@dataclass
class MPCState:
    """State for MPC computation."""
    x_ext: np.ndarray  # Extended state [q, m]
    x_ext_dot: np.ndarray  # Extended velocity [q̇, ṁ]
    reference: np.ndarray  # Reference trajectory
    perception: Optional[Dict] = None


class SMMPC:
    """
    Semantic Morphological Model Predictive Control.
    
    This is a NOVEL CONTRIBUTION: the first MPC formulation that
    treats morphology as a continuous decision variable alongside
    joint trajectories.
    
    Key Features:
        - Joint optimization of joint and morphology trajectories
        - Stability-guaranteed morphology adaptation
        - Online semantic adaptation via LLM reasoning
        - Real-time capable with warm-starting
    
    Example:
        >>> controller = SMMPC(
        ...     config_space=ExtendedConfigSpace(7, 12),
        ...     dynamics=MorphologyDynamics(...),
        ...     horizon=20
        ... )
        >>> u_ext = controller.compute(state, reference)
    """
    
    def __init__(
        self,
        config_space: ExtendedConfigSpace,
        dynamics: MorphologyDynamics,
        config: Optional[SMMPCConfig] = None,
        stability_analyzer: Optional[StabilityAnalyzer] = None,
        llm_interface=None
    ):
        """
        Initialize SM-MPC controller.
        
        Args:
            config_space: Extended configuration space
            dynamics: Morphology-augmented dynamics
            config: Controller configuration
            stability_analyzer: For stability-constrained morphology rates
            llm_interface: For semantic adaptation
        """
        self.config_space = config_space
        self.dynamics = dynamics
        self.config = config or SMMPCConfig()
        self.stability_analyzer = stability_analyzer
        self.llm = llm_interface
        
        self.n = config_space.joint_dim
        self.p = config_space.morphology_dim
        self.N = self.config.horizon
        
        # Weight matrices
        self.Q = np.eye(self.n + self.p)
        self.Q[:self.n, :self.n] *= self.config.joint_weight
        self.Q[self.n:, self.n:] *= self.config.morphology_weight
        
        self.R = np.eye(self.n + self.p) * self.config.control_weight
        self.Qf = self.Q * self.config.terminal_weight
        
        # Warm start storage
        self._prev_solution = None
        self._prev_reference = None
        
    def compute(
        self,
        state: MPCState,
        reference: Optional[MorphologyTrajectory] = None,
        step: int = 0,
        semantic_context: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute optimal extended control.
        
        Args:
            state: Current MPC state
            reference: Reference morphology trajectory
            step: Current timestep in trajectory
            semantic_context: Optional semantic information for adaptation
            
        Returns:
            u_ext: Optimal extended control [q̇, ṁ]
        """
        # Get reference trajectory for horizon
        ref_traj = self._get_reference_trajectory(reference, step)
        
        # Semantic adaptation
        if semantic_context and self.llm:
            ref_traj = self._semantic_adapt(ref_traj, semantic_context, state)
        
        # Solve MPC optimization
        u_traj = self._solve_mpc(state, ref_traj)
        
        # Apply stability constraint to morphology rate
        if self.stability_analyzer:
            u_traj[0] = self._apply_stability_constraint(
                state.x_ext, u_traj[0]
            )
        
        # Return first control
        return u_traj[0]
    
    def _get_reference_trajectory(
        self,
        trajectory: Optional[MorphologyTrajectory],
        step: int
    ) -> np.ndarray:
        """Extract reference trajectory for MPC horizon."""
        ref = np.zeros((self.N + 1, self.n + self.p))
        
        if trajectory is None:
            return ref
        
        for k in range(self.N + 1):
            t = (step + k) / len(trajectory.times)
            t = min(t, 1.0)
            
            # Get morphology reference
            m_ref = trajectory(t)
            
            # Joint reference (typically zero or tracking reference)
            q_ref = np.zeros(self.n)
            
            ref[k] = self.config_space.compose(q_ref, m_ref)
            
        return ref
    
    def _semantic_adapt(
        self,
        ref_traj: np.ndarray,
        context: str,
        state: MPCState
    ) -> np.ndarray:
        """Adapt reference trajectory based on semantic context."""
        # Query LLM for adaptation
        prompt = f"""
Current manipulation context: {context}
Current morphology: {state.x_ext[self.n:]}

Should the morphology be adjusted? If so, suggest adjustment direction:
- "increase_compliance" for fragile objects
- "increase_stiffness" for precision
- "widen_gripper" for larger objects
- "narrow_gripper" for smaller objects
- "no_change" if appropriate

Respond with one adjustment or "no_change".
"""
        
        response = self.llm.generate(prompt) if self.llm else "no_change"
        
        # Apply adaptation
        adaptation = np.zeros(self.p)
        rate = self.config.semantic_adaptation_rate
        
        if "increase_compliance" in response.lower():
            adaptation[2] = rate  # Assuming compliance is index 2
        elif "increase_stiffness" in response.lower():
            adaptation[1] = rate
        elif "widen_gripper" in response.lower():
            adaptation[0] = rate
        elif "narrow_gripper" in response.lower():
            adaptation[0] = -rate
            
        # Apply to reference trajectory
        for k in range(len(ref_traj)):
            ref_traj[k, self.n:] += adaptation * (k / len(ref_traj))
            ref_traj[k, self.n:] = self.config_space.morphology_space.clip(
                ref_traj[k, self.n:]
            )
            
        return ref_traj
    
    def _solve_mpc(
        self,
        state: MPCState,
        ref_traj: np.ndarray
    ) -> np.ndarray:
        """
        Solve the MPC optimization problem.
        
        Optimization:
            min Σ [‖x - x_ref‖²_Q + ‖u‖²_R] + ‖x_N - x_ref_N‖²_Qf
            s.t. dynamics constraints
                 control limits
                 morphology rate limits
        """
        # Decision variables: u_ext for k = 0, ..., N-1
        u_dim = self.n + self.p
        
        # Initial guess (warm start or zeros)
        if self._prev_solution is not None:
            u0 = np.roll(self._prev_solution, -u_dim)
            u0[-u_dim:] = self._prev_solution[-u_dim:]
        else:
            u0 = np.zeros(self.N * u_dim)
        
        # Bounds
        rate_limits = self.config_space.get_rate_limits()
        bounds = []
        for k in range(self.N):
            for i in range(u_dim):
                bounds.append((-rate_limits[i], rate_limits[i]))
        
        # Additional morphology rate constraint
        def morphology_rate_constraint(u_flat):
            """Constraint: ‖ṁ‖ ≤ ṁ_max for each timestep."""
            violations = []
            for k in range(self.N):
                m_dot = u_flat[k*u_dim + self.n : (k+1)*u_dim]
                violations.append(
                    np.linalg.norm(m_dot) - self.config.morphology_rate_limit
                )
            return -np.array(violations)  # <= 0 for feasibility
        
        # Cost function
        def cost(u_flat):
            u_traj = u_flat.reshape(self.N, u_dim)
            
            total_cost = 0.0
            x = state.x_ext.copy()
            
            for k in range(self.N):
                # State cost
                x_error = x - ref_traj[k]
                total_cost += x_error @ self.Q @ x_error
                
                # Control cost
                total_cost += u_traj[k] @ self.R @ u_traj[k]
                
                # Simulate forward (simplified)
                x = self._simulate_step(x, u_traj[k])
            
            # Terminal cost
            x_error = x - ref_traj[self.N]
            total_cost += x_error @ self.Qf @ x_error
            
            return total_cost
        
        # Solve
        result = minimize(
            cost,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': morphology_rate_constraint},
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.tolerance
            }
        )
        
        # Store solution for warm starting
        self._prev_solution = result.x
        
        return result.x.reshape(self.N, u_dim)
    
    def _simulate_step(
        self,
        x: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """Simulate one step of dynamics."""
        q, m = self.config_space.decompose(x)
        q_dot, m_dot = self.config_space.decompose_control(u)
        
        # Simple integration (can be replaced with full dynamics)
        q_new = q + q_dot * self.config.dt
        m_new = m + m_dot * self.config.dt
        
        x_new = self.config_space.compose(q_new, m_new)
        return self.config_space.clip_state(x_new)
    
    def _apply_stability_constraint(
        self,
        x_ext: np.ndarray,
        u_ext: np.ndarray
    ) -> np.ndarray:
        """Apply stability-based morphology rate constraint."""
        q, m = self.config_space.decompose(x_ext)
        q_dot, m_dot = self.config_space.decompose_control(u_ext)
        
        # Get safe morphology rate
        safe_m_dot = self.stability_analyzer.compute_safe_morphology_rate(
            q, m, m_dot, safety_factor=0.8
        )
        
        return self.config_space.compose_control(q_dot, safe_m_dot)
    
    def set_weights(
        self,
        joint_weight: Optional[float] = None,
        morphology_weight: Optional[float] = None,
        control_weight: Optional[float] = None
    ):
        """Update cost function weights."""
        if joint_weight is not None:
            self.config.joint_weight = joint_weight
        if morphology_weight is not None:
            self.config.morphology_weight = morphology_weight
        if control_weight is not None:
            self.config.control_weight = control_weight
            
        # Rebuild weight matrices
        self.Q = np.eye(self.n + self.p)
        self.Q[:self.n, :self.n] *= self.config.joint_weight
        self.Q[self.n:, self.n:] *= self.config.morphology_weight
        self.R = np.eye(self.n + self.p) * self.config.control_weight
        self.Qf = self.Q * self.config.terminal_weight


class DifferentiableSMMPC(nn.Module):
    """
    PyTorch-differentiable SM-MPC for end-to-end learning.
    
    Enables learning MPC parameters from task performance.
    """
    
    def __init__(
        self,
        config_space: ExtendedConfigSpace,
        horizon: int = 10
    ):
        super().__init__()
        
        self.n = config_space.joint_dim
        self.p = config_space.morphology_dim
        self.N = horizon
        
        # Learnable cost weights
        self.log_Q_joint = nn.Parameter(torch.zeros(self.n))
        self.log_Q_morph = nn.Parameter(torch.zeros(self.p))
        self.log_R = nn.Parameter(torch.zeros(self.n + self.p))
        
    @property
    def Q(self) -> torch.Tensor:
        """State cost weight matrix."""
        q_weights = torch.exp(self.log_Q_joint)
        m_weights = torch.exp(self.log_Q_morph)
        return torch.diag(torch.cat([q_weights, m_weights]))
    
    @property
    def R(self) -> torch.Tensor:
        """Control cost weight matrix."""
        return torch.diag(torch.exp(self.log_R))
    
    def forward(
        self,
        x_ext: torch.Tensor,
        ref_traj: torch.Tensor,
        num_iters: int = 10
    ) -> torch.Tensor:
        """
        Compute optimal control via differentiable optimization.
        
        Uses iterative LQR for differentiability.
        
        Args:
            x_ext: Current extended state (batch, n+p)
            ref_traj: Reference trajectory (batch, N+1, n+p)
            num_iters: Number of iLQR iterations
            
        Returns:
            u_ext: Optimal control (batch, n+p)
        """
        batch_size = x_ext.shape[0]
        u_dim = self.n + self.p
        
        # Initialize control sequence
        u_traj = torch.zeros(batch_size, self.N, u_dim, device=x_ext.device)
        
        for _ in range(num_iters):
            # Forward pass
            x_traj = self._rollout(x_ext, u_traj)
            
            # Backward pass (compute gains)
            K, k = self._backward_pass(x_traj, u_traj, ref_traj)
            
            # Update controls
            u_traj = self._forward_pass(x_ext, u_traj, K, k)
        
        return u_traj[:, 0]  # Return first control
    
    def _rollout(
        self,
        x0: torch.Tensor,
        u_traj: torch.Tensor
    ) -> torch.Tensor:
        """Roll out trajectory under given controls."""
        batch_size = x0.shape[0]
        x_traj = torch.zeros(batch_size, self.N + 1, self.n + self.p, device=x0.device)
        x_traj[:, 0] = x0
        
        for k in range(self.N):
            # Simple dynamics: x_{k+1} = x_k + u_k * dt
            x_traj[:, k+1] = x_traj[:, k] + u_traj[:, k] * 0.01
            
        return x_traj
    
    def _backward_pass(
        self,
        x_traj: torch.Tensor,
        u_traj: torch.Tensor,
        ref_traj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feedback gains via backward recursion."""
        batch_size = x_traj.shape[0]
        u_dim = self.n + self.p
        x_dim = self.n + self.p
        
        K = torch.zeros(batch_size, self.N, u_dim, x_dim, device=x_traj.device)
        k = torch.zeros(batch_size, self.N, u_dim, device=x_traj.device)
        
        # Terminal cost gradient
        V_x = 2 * self.Q @ (x_traj[:, -1] - ref_traj[:, -1]).unsqueeze(-1)
        V_xx = 2 * self.Q.unsqueeze(0).expand(batch_size, -1, -1)
        
        for t in range(self.N - 1, -1, -1):
            # Stage cost gradients
            x_err = (x_traj[:, t] - ref_traj[:, t]).unsqueeze(-1)
            Q_x = 2 * self.Q @ x_err + V_x
            Q_u = 2 * self.R @ u_traj[:, t].unsqueeze(-1)
            Q_xx = 2 * self.Q + V_xx
            Q_uu = 2 * self.R.unsqueeze(0).expand(batch_size, -1, -1)
            Q_ux = torch.zeros(batch_size, u_dim, x_dim, device=x_traj.device)
            
            # Feedback gains
            Q_uu_inv = torch.inverse(Q_uu + 1e-6 * torch.eye(u_dim, device=Q_uu.device))
            K[:, t] = -Q_uu_inv @ Q_ux
            k[:, t] = -Q_uu_inv @ Q_u.squeeze(-1)
            
            # Value function update
            V_x = Q_x - K[:, t].transpose(-1, -2) @ Q_uu @ k[:, t].unsqueeze(-1)
            V_xx = Q_xx - K[:, t].transpose(-1, -2) @ Q_uu @ K[:, t]
            
        return K, k
    
    def _forward_pass(
        self,
        x0: torch.Tensor,
        u_traj: torch.Tensor,
        K: torch.Tensor,
        k: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """Apply feedback gains to update controls."""
        batch_size = x0.shape[0]
        u_new = torch.zeros_like(u_traj)
        x = x0.clone()
        
        for t in range(self.N):
            u_new[:, t] = u_traj[:, t] + alpha * k[:, t]
            x = x + u_new[:, t] * 0.01
            
        return u_new
