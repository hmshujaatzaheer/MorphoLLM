"""
SM-MPC Control Demo

This example demonstrates the Semantic Morphological Model Predictive
Control (SM-MPC) for real-time morphology adaptation.

Author: H M Shujaat Zaheer
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from morphollm import (
    ExtendedConfigSpace,
    MorphologySpace,
    SMMPC,
    SMMPCConfig,
    L2MT
)
from morphollm.core.morphology_dynamics import MorphologyDynamics
from morphollm.core.stability_analysis import StabilityAnalyzer
from morphollm.algorithms.sm_mpc import MPCState


@dataclass
class SimulatedRobot:
    """Simple simulated robot for demo."""
    x_ext: np.ndarray
    x_ext_dot: np.ndarray
    config_space: ExtendedConfigSpace
    dt: float = 0.01
    
    def step(self, u_ext: np.ndarray):
        """Apply control and update state."""
        # Clip control
        u_ext = self.config_space.clip_control(u_ext)
        
        # Simple dynamics: x_dot = u
        self.x_ext_dot = u_ext
        self.x_ext = self.x_ext + u_ext * self.dt
        self.x_ext = self.config_space.clip_state(self.x_ext)
        
        return self.x_ext.copy()
    
    def get_extended_state(self) -> np.ndarray:
        return self.x_ext.copy()


def main():
    """Run SM-MPC demonstration."""
    
    print("=" * 60)
    print("MorphoLLM: Semantic Morphological MPC Demo")
    print("=" * 60)
    
    # 1. Setup configuration space
    print("\n1. Setting up extended configuration space...")
    config_space = ExtendedConfigSpace(
        joint_dim=3,
        morphology_dim=4
    )
    print(f"   Joint dim: {config_space.joint_dim}")
    print(f"   Morphology dim: {config_space.morphology_dim}")
    print(f"   Total dim: {config_space.total_dim}")
    
    # 2. Create dynamics model
    print("\n2. Creating morphology-augmented dynamics...")
    dynamics = MorphologyDynamics(config_space)
    
    # 3. Create stability analyzer
    print("\n3. Initializing stability analyzer...")
    damping_gain = np.eye(config_space.joint_dim) * 15.0
    stability_analyzer = StabilityAnalyzer(dynamics, damping_gain)
    
    # 4. Configure SM-MPC
    print("\n4. Configuring SM-MPC controller...")
    mpc_config = SMMPCConfig(
        horizon=10,
        dt=0.02,
        joint_weight=1.0,
        morphology_weight=0.5,
        control_weight=0.01,
        morphology_rate_limit=0.15
    )
    
    controller = SMMPC(
        config_space=config_space,
        dynamics=dynamics,
        config=mpc_config,
        stability_analyzer=stability_analyzer
    )
    print(f"   Horizon: {mpc_config.horizon}")
    print(f"   Morphology rate limit: {mpc_config.morphology_rate_limit}")
    
    # 5. Generate reference trajectory using L2MT
    print("\n5. Generating reference trajectory...")
    l2mt = L2MT(
        morphology_space=config_space.morphology_space,
        llm_backend="mock"
    )
    
    task = "Grasp a delicate object and place it precisely"
    initial_morph = np.array([0.5, 0.5, 0.3, 0.5])
    
    ref_trajectory = l2mt.generate(
        task_description=task,
        initial_morphology=initial_morph,
        horizon=200
    )
    print(f"   Task: '{task}'")
    print(f"   Reference duration: {len(ref_trajectory.times)} steps")
    
    # 6. Initialize robot
    print("\n6. Initializing simulated robot...")
    initial_q = np.zeros(config_space.joint_dim)
    robot = SimulatedRobot(
        x_ext=config_space.compose(initial_q, initial_morph),
        x_ext_dot=np.zeros(config_space.total_dim),
        config_space=config_space
    )
    
    # 7. Run control loop
    print("\n7. Running SM-MPC control loop...")
    num_steps = 150
    
    # Storage for plotting
    states = [robot.get_extended_state()]
    controls = []
    morphology_rates = []
    stability_margins = []
    
    for step in range(num_steps):
        # Create MPC state
        mpc_state = MPCState(
            x_ext=robot.x_ext,
            x_ext_dot=robot.x_ext_dot,
            reference=ref_trajectory.morphologies
        )
        
        # Compute optimal control
        u_ext = controller.compute(
            state=mpc_state,
            reference=ref_trajectory,
            step=step
        )
        
        # Check stability
        q, m = config_space.decompose(robot.x_ext)
        _, m_dot = config_space.decompose_control(u_ext)
        bounds = stability_analyzer.compute_stability_bounds(
            q, np.zeros_like(q), m, m_dot
        )
        stability_margins.append(bounds.stability_margin)
        
        # Apply control
        robot.step(u_ext)
        
        # Store for plotting
        states.append(robot.get_extended_state())
        controls.append(u_ext)
        morphology_rates.append(np.linalg.norm(m_dot))
        
        if step % 30 == 0:
            print(f"   Step {step}: m_rate={np.linalg.norm(m_dot):.4f}, "
                  f"stability_margin={bounds.stability_margin:.4f}")
    
    print("\n   Control loop complete!")
    
    # 8. Analyze results
    states = np.array(states)
    controls = np.array(controls)
    
    print("\n8. Results summary:")
    print(f"   Total steps: {num_steps}")
    print(f"   Max morphology rate: {max(morphology_rates):.4f}")
    print(f"   Min stability margin: {min(stability_margins):.4f}")
    print(f"   All steps stable: {all(m > 0 for m in stability_margins)}")
    
    # 9. Visualize
    print("\n9. Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Joint trajectories
    ax1 = axes[0, 0]
    for i in range(config_space.joint_dim):
        ax1.plot(states[:, i], label=f'Joint {i+1}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Joint Position')
    ax1.set_title('Joint Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Morphology trajectories
    ax2 = axes[0, 1]
    morph_names = ['Width', 'Stiffness', 'Compliance', 'Param4']
    for i in range(config_space.morphology_dim):
        ax2.plot(states[:, config_space.joint_dim + i], 
                label=morph_names[i] if i < len(morph_names) else f'M{i+1}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Morphology Value')
    ax2.set_title('Morphology Adaptation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Morphology rate
    ax3 = axes[1, 0]
    ax3.plot(morphology_rates, 'b-', label='Actual rate')
    ax3.axhline(y=mpc_config.morphology_rate_limit, color='r', 
               linestyle='--', label='Rate limit')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('||ṁ||')
    ax3.set_title('Morphology Rate (Stability Constrained)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Stability margin
    ax4 = axes[1, 1]
    ax4.plot(stability_margins, 'g-')
    ax4.axhline(y=0, color='r', linestyle='--', label='Stability boundary')
    ax4.fill_between(range(len(stability_margins)), 0, stability_margins,
                     where=np.array(stability_margins) > 0, 
                     alpha=0.3, color='green', label='Stable region')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Margin')
    ax4.set_title('Stability Margin (Lyapunov-based)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sm_mpc_demo_output.png', dpi=150)
    print("   Saved visualization to 'sm_mpc_demo_output.png'")
    
    print("\n" + "=" * 60)
    print("SM-MPC Demo completed successfully!")
    print("=" * 60)
    
    return states, controls


if __name__ == "__main__":
    states, controls = main()
