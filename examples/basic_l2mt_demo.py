"""
Basic L2MT Demo

This example demonstrates the core functionality of the
Language-to-Morphology-Trajectory (L2MT) algorithm.

Author: H M Shujaat Zaheer
"""

import numpy as np
import matplotlib.pyplot as plt

from morphollm import L2MT, MorphologySpace, ExtendedConfigSpace


def main():
    """Run basic L2MT demonstration."""
    
    print("=" * 60)
    print("MorphoLLM: Language-to-Morphology-Trajectory Demo")
    print("=" * 60)
    
    # 1. Define morphology space
    print("\n1. Defining morphology space...")
    morphology_space = MorphologySpace(
        dim=6,
        parameter_names=[
            "gripper_width",
            "stiffness",
            "compliance",
            "wrist_angle",
            "finger_spread",
            "palm_curvature"
        ],
        bounds=(np.zeros(6), np.ones(6)),
        rate_limits=np.ones(6) * 0.15
    )
    print(f"   Morphology dimension: {morphology_space.dim}")
    print(f"   Parameters: {morphology_space.parameter_names}")
    
    # 2. Initialize L2MT
    print("\n2. Initializing L2MT algorithm...")
    l2mt = L2MT(
        morphology_space=morphology_space,
        llm_backend="mock",  # Use mock for demo
        physics_validator=True
    )
    print("   L2MT initialized with mock LLM backend")
    
    # 3. Define a manipulation task
    task = "Pick up the fragile glass vase carefully and place it on the shelf"
    print(f"\n3. Task: '{task}'")
    
    # 4. Generate morphology trajectory
    print("\n4. Generating morphology trajectory...")
    initial_morphology = np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.5])
    
    trajectory = l2mt.generate(
        task_description=task,
        initial_morphology=initial_morphology,
        horizon=100,
        physics_constraints={"fragile": True}
    )
    
    print(f"   Generated trajectory with {len(trajectory.times)} timesteps")
    print(f"   Duration: {trajectory.duration:.2f} (normalized)")
    print(f"   Number of phases: {len(trajectory.phases)}")
    
    # 5. Display phases
    print("\n5. Task phases:")
    for phase in trajectory.phases:
        print(f"   [{phase.start_time:.2f}-{phase.end_time:.2f}] {phase.description}")
        print(f"      Tags: {phase.semantic_tags}")
    
    # 6. Sample trajectory at different times
    print("\n6. Morphology at key timepoints:")
    for t in [0.0, 0.3, 0.6, 1.0]:
        m = trajectory(t)
        print(f"   t={t:.1f}: gripper={m[0]:.2f}, stiffness={m[1]:.2f}, compliance={m[2]:.2f}")
    
    # 7. Compute morphology rates
    print("\n7. Maximum morphology rate:")
    m_dot = np.diff(trajectory.morphologies, axis=0) / np.diff(trajectory.times)[:, None]
    max_rate = np.max(np.abs(m_dot))
    print(f"   Max |ṁ|: {max_rate:.4f}")
    
    # 8. Visualize (if matplotlib available)
    try:
        print("\n8. Visualizing trajectory...")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # Morphology over time
        ax1 = axes[0]
        for i in range(3):
            ax1.plot(trajectory.times, trajectory.morphologies[:, i],
                    label=morphology_space.parameter_names[i])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title('Morphology Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phase boundaries
        ax2 = axes[1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(trajectory.phases)))
        for phase, color in zip(trajectory.phases, colors):
            ax2.axvspan(phase.start_time, phase.end_time, 
                       alpha=0.5, color=color, label=phase.description[:30])
        ax2.set_xlabel('Time')
        ax2.set_title('Task Phases')
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('l2mt_demo_output.png', dpi=150)
        print("   Saved visualization to 'l2mt_demo_output.png'")
        
    except Exception as e:
        print(f"   Visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    return trajectory


if __name__ == "__main__":
    trajectory = main()
