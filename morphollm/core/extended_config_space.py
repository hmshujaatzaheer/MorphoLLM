"""
Extended Configuration Space for MorphoLLM.

This module implements the mathematical foundation for treating morphology
as a continuous control variable alongside joint configurations.

Mathematical Foundation:
    C_ext = C × M
    
    where:
    - C ⊆ R^n is the joint configuration space
    - M ⊆ R^p is the morphology space
    - C_ext is the extended configuration space

Author: H M Shujaat Zaheer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Callable
import torch
import torch.nn as nn


@dataclass
class MorphologySpace:
    """
    Defines the morphology parameter space M ⊆ R^p.
    
    Morphology parameters can include:
    - Link lengths
    - Stiffness distributions  
    - Gripper geometry
    - Compliance parameters
    
    Attributes:
        dim: Dimension of morphology space (p)
        bounds: (lower, upper) bounds for each dimension
        parameter_names: Names of morphology parameters
        rate_limits: Maximum rate of change for each parameter
    """
    dim: int
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    parameter_names: Optional[List[str]] = None
    rate_limits: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.bounds is None:
            # Default bounds: [0, 1] for all parameters
            self.bounds = (
                np.zeros(self.dim),
                np.ones(self.dim)
            )
        
        if self.parameter_names is None:
            self.parameter_names = [f"m_{i}" for i in range(self.dim)]
            
        if self.rate_limits is None:
            # Default: 0.1 units per timestep
            self.rate_limits = np.ones(self.dim) * 0.1
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample random morphology configurations."""
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.dim))
    
    def clip(self, m: np.ndarray) -> np.ndarray:
        """Clip morphology to valid bounds."""
        lower, upper = self.bounds
        return np.clip(m, lower, upper)
    
    def clip_rate(self, m_dot: np.ndarray) -> np.ndarray:
        """Clip morphology rate to valid limits."""
        return np.clip(m_dot, -self.rate_limits, self.rate_limits)
    
    def is_valid(self, m: np.ndarray) -> bool:
        """Check if morphology is within bounds."""
        lower, upper = self.bounds
        return np.all(m >= lower) and np.all(m <= upper)
    
    def distance(self, m1: np.ndarray, m2: np.ndarray) -> float:
        """Compute distance between morphologies."""
        return np.linalg.norm(m1 - m2)


@dataclass 
class JointSpace:
    """
    Defines the joint configuration space C ⊆ R^n.
    
    Attributes:
        dim: Number of joints (n)
        joint_limits: (lower, upper) position limits
        velocity_limits: Maximum joint velocities
        joint_names: Names of joints
        joint_types: 'revolute' or 'prismatic' for each joint
    """
    dim: int
    joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    velocity_limits: Optional[np.ndarray] = None
    joint_names: Optional[List[str]] = None
    joint_types: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.joint_limits is None:
            # Default: [-π, π] for all joints
            self.joint_limits = (
                np.full(self.dim, -np.pi),
                np.full(self.dim, np.pi)
            )
            
        if self.velocity_limits is None:
            self.velocity_limits = np.ones(self.dim) * 2.0  # rad/s
            
        if self.joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(self.dim)]
            
        if self.joint_types is None:
            self.joint_types = ['revolute'] * self.dim
    
    def clip(self, q: np.ndarray) -> np.ndarray:
        """Clip joint configuration to limits."""
        lower, upper = self.joint_limits
        return np.clip(q, lower, upper)
    
    def clip_velocity(self, q_dot: np.ndarray) -> np.ndarray:
        """Clip joint velocities to limits."""
        return np.clip(q_dot, -self.velocity_limits, self.velocity_limits)


class ExtendedConfigSpace:
    """
    Extended Configuration Space C_ext = C × M.
    
    This class unifies joint configurations and morphology parameters
    into a single mathematical framework for planning and control.
    
    The extended state is:
        x_ext = [q, m]^T ∈ R^(n+p)
        
    The extended control is:
        u_ext = [q_dot, m_dot]^T ∈ R^(n+p)
    
    Example:
        >>> C_ext = ExtendedConfigSpace(joint_dim=7, morphology_dim=12)
        >>> x = C_ext.compose(q=joint_config, m=morphology)
        >>> q, m = C_ext.decompose(x)
    """
    
    def __init__(
        self,
        joint_dim: int,
        morphology_dim: int,
        joint_space: Optional[JointSpace] = None,
        morphology_space: Optional[MorphologySpace] = None
    ):
        """
        Initialize the Extended Configuration Space.
        
        Args:
            joint_dim: Dimension of joint space (n)
            morphology_dim: Dimension of morphology space (p)
            joint_space: Optional pre-configured JointSpace
            morphology_space: Optional pre-configured MorphologySpace
        """
        self.joint_dim = joint_dim
        self.morphology_dim = morphology_dim
        self.total_dim = joint_dim + morphology_dim
        
        self.joint_space = joint_space or JointSpace(dim=joint_dim)
        self.morphology_space = morphology_space or MorphologySpace(dim=morphology_dim)
        
        # Validate dimensions
        assert self.joint_space.dim == joint_dim
        assert self.morphology_space.dim == morphology_dim
        
    def compose(
        self, 
        q: np.ndarray, 
        m: np.ndarray
    ) -> np.ndarray:
        """
        Compose extended state from joint config and morphology.
        
        Args:
            q: Joint configuration (n,)
            m: Morphology parameters (p,)
            
        Returns:
            x_ext: Extended state (n+p,)
        """
        return np.concatenate([q, m])
    
    def decompose(
        self, 
        x_ext: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose extended state into joint config and morphology.
        
        Args:
            x_ext: Extended state (n+p,)
            
        Returns:
            q: Joint configuration (n,)
            m: Morphology parameters (p,)
        """
        q = x_ext[:self.joint_dim]
        m = x_ext[self.joint_dim:]
        return q, m
    
    def compose_control(
        self,
        q_dot: np.ndarray,
        m_dot: np.ndarray
    ) -> np.ndarray:
        """
        Compose extended control from joint velocities and morphology rates.
        
        Args:
            q_dot: Joint velocities (n,)
            m_dot: Morphology rates (p,)
            
        Returns:
            u_ext: Extended control (n+p,)
        """
        return np.concatenate([q_dot, m_dot])
    
    def decompose_control(
        self,
        u_ext: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose extended control into joint velocities and morphology rates.
        
        Args:
            u_ext: Extended control (n+p,)
            
        Returns:
            q_dot: Joint velocities (n,)
            m_dot: Morphology rates (p,)
        """
        q_dot = u_ext[:self.joint_dim]
        m_dot = u_ext[self.joint_dim:]
        return q_dot, m_dot
    
    def clip_state(self, x_ext: np.ndarray) -> np.ndarray:
        """Clip extended state to valid bounds."""
        q, m = self.decompose(x_ext)
        q_clipped = self.joint_space.clip(q)
        m_clipped = self.morphology_space.clip(m)
        return self.compose(q_clipped, m_clipped)
    
    def clip_control(self, u_ext: np.ndarray) -> np.ndarray:
        """Clip extended control to valid limits."""
        q_dot, m_dot = self.decompose_control(u_ext)
        q_dot_clipped = self.joint_space.clip_velocity(q_dot)
        m_dot_clipped = self.morphology_space.clip_rate(m_dot)
        return self.compose_control(q_dot_clipped, m_dot_clipped)
    
    def sample_state(self, n: int = 1) -> np.ndarray:
        """Sample random extended states."""
        q = self.joint_space.clip(np.random.randn(n, self.joint_dim))
        m = self.morphology_space.sample(n)
        if n == 1:
            return self.compose(q[0], m[0])
        return np.hstack([q, m])
    
    def distance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        weights: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Compute weighted distance in extended space.
        
        Args:
            x1: First extended state
            x2: Second extended state  
            weights: (joint_weight, morphology_weight)
            
        Returns:
            Weighted distance
        """
        weights = weights or (1.0, 1.0)
        q1, m1 = self.decompose(x1)
        q2, m2 = self.decompose(x2)
        
        q_dist = np.linalg.norm(q1 - q2)
        m_dist = self.morphology_space.distance(m1, m2)
        
        return weights[0] * q_dist + weights[1] * m_dist
    
    def interpolate(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Linear interpolation in extended space.
        
        Args:
            x1: Start state
            x2: End state
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated state
        """
        return (1 - t) * x1 + t * x2
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for the extended configuration space."""
        joint_lower, joint_upper = self.joint_space.joint_limits
        morph_lower, morph_upper = self.morphology_space.bounds
        
        lower = np.concatenate([joint_lower, morph_lower])
        upper = np.concatenate([joint_upper, morph_upper])
        
        return lower, upper
    
    def get_rate_limits(self) -> np.ndarray:
        """Get rate limits for extended control."""
        return np.concatenate([
            self.joint_space.velocity_limits,
            self.morphology_space.rate_limits
        ])


class ExtendedConfigSpaceTorch(nn.Module):
    """
    PyTorch-compatible Extended Configuration Space for differentiable operations.
    """
    
    def __init__(self, config_space: ExtendedConfigSpace):
        super().__init__()
        self.config_space = config_space
        
        # Register bounds as buffers
        lower, upper = config_space.get_bounds()
        self.register_buffer('lower_bounds', torch.tensor(lower, dtype=torch.float32))
        self.register_buffer('upper_bounds', torch.tensor(upper, dtype=torch.float32))
        self.register_buffer('rate_limits', torch.tensor(
            config_space.get_rate_limits(), dtype=torch.float32
        ))
        
    def compose(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compose extended state (differentiable)."""
        return torch.cat([q, m], dim=-1)
    
    def decompose(self, x_ext: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose extended state (differentiable)."""
        n = self.config_space.joint_dim
        return x_ext[..., :n], x_ext[..., n:]
    
    def clip_state(self, x_ext: torch.Tensor) -> torch.Tensor:
        """Clip state with smooth clamping."""
        return torch.clamp(x_ext, self.lower_bounds, self.upper_bounds)
    
    def clip_control(self, u_ext: torch.Tensor) -> torch.Tensor:
        """Clip control with smooth clamping."""
        return torch.clamp(u_ext, -self.rate_limits, self.rate_limits)


# Convenience functions
def create_manipulator_space(
    num_joints: int = 7,
    gripper_params: int = 6,
    arm_compliance: int = 3,
    wrist_params: int = 3
) -> ExtendedConfigSpace:
    """
    Create a standard manipulator extended configuration space.
    
    Args:
        num_joints: Number of arm joints
        gripper_params: Number of gripper morphology parameters
        arm_compliance: Number of arm compliance parameters
        wrist_params: Number of wrist morphology parameters
        
    Returns:
        Configured ExtendedConfigSpace
    """
    morphology_dim = gripper_params + arm_compliance + wrist_params
    
    morph_names = (
        [f"gripper_{i}" for i in range(gripper_params)] +
        [f"compliance_{i}" for i in range(arm_compliance)] +
        [f"wrist_{i}" for i in range(wrist_params)]
    )
    
    morphology_space = MorphologySpace(
        dim=morphology_dim,
        parameter_names=morph_names
    )
    
    return ExtendedConfigSpace(
        joint_dim=num_joints,
        morphology_dim=morphology_dim,
        morphology_space=morphology_space
    )
