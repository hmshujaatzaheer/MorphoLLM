"""
Full MorphoLLM Pipeline Demo

This example demonstrates the complete MorphoLLM pipeline:
1. Task specification in natural language
2. L2MT trajectory generation
3. SM-MPC control execution
4. Stability-guaranteed morphology adaptation

Author: H M Shujaat Zaheer
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# Import MorphoLLM components
from morphollm import (
    ExtendedConfigSpace,
    MorphologySpace,
    MorphologyDynamics,
    StabilityAnalyzer,
    L2MT,
    SMMPC,
    SMMPCConfig,
)


@dataclass
class SimulatedRobot:
    """Simple simulated robot for demonstration."""
    
    config_space: ExtendedConfigSpace
    state: np.ndarray  # [q, m]
    velocity: np.ndarray  # [q_dot, m_dot]
    
    def __post_init__(self):
        self.task_complete_flag = False
        self.step_count = 0
        self.max_steps = 200
        self.target_position = np.array([1.0, 0.5, 0.3])
        
    def get_extended_state(self) -> np.ndarray:
        return self.state.copy()
    
    def get_morphology(self) -> np.ndarray:
        _, m = self.config_space.decompose(self.state)
        return m
    
    def get_perception(self) -> dict:
        return {
            "object_detected": True,
            "object_fragility": 0.8,
            "distance_to_target": np.linalg.norm(
                self.state[:3] - self.target_position
            )
        }
    
    def apply_control(self, u_ext: np.ndarray, dt: float = 0.01):
        """Apply extended control to robot."""
        # Clip control
        u_ext = self.config_space.clip_control(u_ext)
        
        # Simple integration
        self.velocity = u_ext
        self.state = self.state + u_ext * dt
        self.state = self.config_space.clip_state(self.state)
        
        self.step_count += 1
        
        # Check task completion
        q, m = self.config_space.decompose(self.state)
        if np.linalg.norm(q[:3] - self.target_position) < 0.1:
            self.task_complete_flag = True
    
    def task_complete(self) -> bool:
        return self.task_complete_flag or self.step_count >= self.max_steps
    
    def render(self):
        """Placeholder for visualization."""
        pass


def run_full_pipeline():
    """Run the complete MorphoLLM pipeline."""
    
    print("=" * 70)
    print("MorphoLLM: Full Pipeline Demonstration")
    print("Language-Guided Morphological Trajectory Synthesis")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Setup Configuration Spaces
    # =========================================================================
    print("\n[Step 1] Setting up configuration spaces...")
    
    joint_dim = 7  # 7-DOF manipulator
    morphology_dim = 6  # 6 morphology parameters
    
    morphology_space = MorphologySpace(
        dim=morphology_dim,
        parameter_names=[
            "gripper_width", "stiffness", "compliance",
            "wrist_angle", "finger_spread", "palm_curvature"
        ],
        rate_limits=np.ones(morphology_dim) * 0.1
    )
    
    config_space = ExtendedConfigSpace(
        joint_dim=joint_dim,
        morphology_dim=morphology_dim,
        morphology_space=morphology_space
    )
    
    print(f"   Joint dimensions: {joint_dim}")
    print(f"   Morphology dimensions: {morphology_dim}")
    print(f"   Extended space dimension: {config_space.total_dim}")
    
    # =========================================================================
    # STEP 2: Initialize Dynamics and Stability Analyzer
    # =========================================================================
    print("\n[Step 2] Initializing dynamics and stability analyzer...")
    
    dynamics = MorphologyDynamics(config_space)
    
    damping_gain = np.eye(joint_dim) * 15.0
    stability_analyzer = StabilityAnalyzer(dynamics, damping_gain)
    
    print(f"   Dynamics model initialized")
    print(f"   Stability analyzer configured with damping λ_min = {np.min(np.linalg.eigvalsh(damping_gain)):.1f}")
    
    # =========================================================================
    # STEP 3: Initialize L2MT Algorithm
    # =========================================================================
    print("\n[Step 3] Initializing L2MT algorithm...")
    
    l2mt = L2MT(
        morphology_space=morphology_space,
        llm_backend="mock",  # Use mock for demo
        physics_validator=True
    )
    
    print(f"   L2MT initialized with mock LLM backend")
    
    # =========================================================================
    # STEP 4: Define Task and Generate Morphology Trajectory
    # =========================================================================
    print("\n[Step 4] Generating morphology trajectory from task...")
    
    task = "Carefully pick up the fragile glass vase and place it gently on the upper shelf"
    print(f"   Task: '{task}'")
    
    initial_morphology = np.array([0.6, 0.5, 0.3, 0.5, 0.5, 0.5])
    
    trajectory = l2mt.generate(
        task_description=task,
        initial_morphology=initial_morphology,
        horizon=200,
        physics_constraints={"fragile": True, "max_force": 5.0}
    )
    
    print(f"   Generated {len(trajectory.times)} timesteps")
    print(f"   Task phases:")
    for phase in trajectory.phases:
        print(f"      [{phase.start_time:.2f}-{phase.end_time:.2f}] {phase.description}")
    
    # =========================================================================
    # STEP 5: Compute Stability Bounds
    # =========================================================================
    print("\n[Step 5] Computing stability bounds...")
    
    q_test = np.zeros(joint_dim)
    m_test = initial_morphology
    
    bounds = stability_analyzer.compute_stability_bounds(
        q_test, np.zeros(joint_dim), m_test
    )
    
    print(f"   Maximum safe morphology rate: {bounds.max_morphology_rate:.4f}")
    print(f"   System stable: {bounds.is_stable}")
    
    # =========================================================================
    # STEP 6: Initialize SM-MPC Controller
    # =========================================================================
    print("\n[Step 6] Initializing SM-MPC controller...")
    
    mpc_config = SMMPCConfig(
        horizon=20,
        dt=0.01,
        joint_weight=1.0,
        morphology_weight=0.5,
        control_weight=0.01,
        morphology_rate_limit=min(0.1, bounds.max_morphology_rate * 0.8)
    )
    
    controller = SMMPC(
        config_space=config_space,
        dynamics=dynamics,
        config=mpc_config,
        stability_analyzer=stability_analyzer
    )
    
    print(f"   MPC horizon: {mpc_config.horizon}")
    print(f"   Morphology rate limit: {mpc_config.morphology_rate_limit:.4f}")
    
    # =========================================================================
    # STEP 7: Initialize Robot and Run Control Loop
    # =========================================================================
    print("\n[Step 7] Running control loop...")
    
    initial_q = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
    initial_state = config_space.compose(initial_q, initial_morphology)
    
    robot = SimulatedRobot(
        config_space=config_space,
        state=initial_state,
        velocity=np.zeros(config_space.total_dim)
    )
    
    # Storage for logging
    state_log = [robot.state.copy()]
    control_log = []
    stability_log = []
    
    step = 0
    adaptation_count = 0
    
    from morphollm.algorithms.sm_mpc import MPCState
    
    while not robot.task_complete():
        # Get current state
        x_ext = robot.get_extended_state()
        q, m = config_space.decompose(x_ext)
        
        # Create MPC state
        mpc_state = MPCState(
            x_ext=x_ext,
            x_ext_dot=robot.velocity,
            reference=trajectory.morphologies
        )
        
        # Compute control
        u_ext = controller.compute(
            state=mpc_state,
            reference=trajectory,
            step=step
        )
        
        # Check stability
        q_dot, m_dot = config_space.decompose_control(u_ext)
        stability_bounds = stability_analyzer.compute_stability_bounds(
            q, q_dot, m, m_dot
        )
        
        # Apply stability-safe control
        if not stability_bounds.is_stable:
            safe_m_dot = stability_analyzer.compute_safe_morphology_rate(
                q, m, m_dot, safety_factor=0.7
            )
            u_ext = config_space.compose_control(q_dot, safe_m_dot)
            adaptation_count += 1
        
        # Apply control
        robot.apply_control(u_ext)
        
        # Log
        state_log.append(robot.state.copy())
        control_log.append(u_ext.copy())
        stability_log.append(stability_bounds.stability_margin)
        
        step += 1
        
        # Progress update
        if step % 50 == 0:
            print(f"   Step {step}: position error = {robot.get_perception()['distance_to_target']:.3f}")
    
    print(f"\n   Control loop completed in {step} steps")
    print(f"   Stability adaptations: {adaptation_count}")
    print(f"   Task success: {robot.task_complete_flag}")
    
    # =========================================================================
    # STEP 8: Visualize Results
    # =========================================================================
    print("\n[Step 8] Generating visualizations...")
    
    state_log = np.array(state_log)
    control_log = np.array(control_log) if control_log else np.zeros((1, config_space.total_dim))
    stability_log = np.array(stability_log) if stability_log else np.zeros(1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Joint trajectory
    ax1 = axes[0, 0]
    times = np.arange(len(state_log)) * mpc_config.dt
    for i in range(min(3, joint_dim)):
        ax1.plot(times, state_log[:, i], label=f'q_{i}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint Position (rad)')
    ax1.set_title('Joint Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Morphology trajectory
    ax2 = axes[0, 1]
    for i in range(min(3, morphology_dim)):
        ax2.plot(times, state_log[:, joint_dim + i], 
                label=morphology_space.parameter_names[i])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Morphology Value')
    ax2.set_title('Morphology Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability margin
    ax3 = axes[1, 0]
    if len(stability_log) > 0:
        ax3.plot(times[1:len(stability_log)+1], stability_log, 'b-', linewidth=1.5)
        ax3.axhline(y=0, color='r', linestyle='--', label='Stability boundary')
        ax3.fill_between(times[1:len(stability_log)+1], 0, stability_log, 
                        where=stability_log > 0, alpha=0.3, color='green')
        ax3.fill_between(times[1:len(stability_log)+1], 0, stability_log,
                        where=stability_log <= 0, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Stability Margin')
    ax3.set_title('Morphological Stability Margin')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Task phases
    ax4 = axes[1, 1]
    colors = plt.cm.Set2(np.linspace(0, 1, len(trajectory.phases)))
    for phase, color in zip(trajectory.phases, colors):
        t_start = phase.start_time * times[-1]
        t_end = phase.end_time * times[-1]
        ax4.axvspan(t_start, t_end, alpha=0.4, color=color, 
                   label=phase.description[:25])
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Task Phases')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_xlim(0, times[-1])
    
    plt.tight_layout()
    plt.savefig('full_pipeline_demo.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'full_pipeline_demo.png'")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Duration: {step * mpc_config.dt:.2f} seconds")
    print(f"Total steps: {step}")
    print(f"Success: {robot.task_complete_flag}")
    print(f"Stability margin (mean): {np.mean(stability_log):.4f}")
    print(f"Stability adaptations: {adaptation_count}")
    print("=" * 70)
    
    return {
        'success': robot.task_complete_flag,
        'steps': step,
        'state_log': state_log,
        'trajectory': trajectory
    }


if __name__ == "__main__":
    results = run_full_pipeline()
