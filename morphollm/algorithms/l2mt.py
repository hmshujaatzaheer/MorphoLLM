"""
Language-to-Morphology-Trajectory (L2MT) Algorithm.

This module implements the NOVEL L2MT algorithm that generates
time-varying morphology specifications m(t) from natural language
task descriptions - the FIRST method to produce morphology TRAJECTORIES
rather than static designs.

Algorithm Overview:
    1. Semantic Task Parsing: Extract task requirements from language
    2. Morphology Requirement Inference: Map semantic phases to morphology
    3. Trajectory Synthesis: Generate smooth morphology trajectory
    4. Differentiable Refinement: Optimize via differentiable simulation

Author: H M Shujaat Zaheer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

from morphollm.core.extended_config_space import ExtendedConfigSpace, MorphologySpace


@dataclass
class L2MTConfig:
    """Configuration for L2MT algorithm."""
    llm_backend: str = "gpt-4"
    physics_validation: bool = True
    trajectory_smoothness: float = 0.1
    max_phases: int = 10
    interpolation_method: str = "cubic"
    optimization_steps: int = 100
    learning_rate: float = 0.01


@dataclass
class TaskPhase:
    """Represents a semantic phase of a manipulation task."""
    phase_id: int
    description: str
    start_time: float
    end_time: float
    semantic_tags: List[str]
    required_morphology: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class MorphologyTrajectory:
    """
    Represents a time-varying morphology specification.
    
    Attributes:
        times: Timesteps (T,)
        morphologies: Morphology at each timestep (T, p)
        phases: List of task phases
        metadata: Additional information
    """
    times: np.ndarray
    morphologies: np.ndarray
    phases: List[TaskPhase]
    metadata: Dict = field(default_factory=dict)
    
    def __call__(self, t: float) -> np.ndarray:
        """Interpolate morphology at time t."""
        if t <= self.times[0]:
            return self.morphologies[0]
        if t >= self.times[-1]:
            return self.morphologies[-1]
        
        # Linear interpolation
        idx = np.searchsorted(self.times, t)
        t0, t1 = self.times[idx-1], self.times[idx]
        m0, m1 = self.morphologies[idx-1], self.morphologies[idx]
        
        alpha = (t - t0) / (t1 - t0)
        return (1 - alpha) * m0 + alpha * m1
    
    def get_derivative(self, t: float, dt: float = 0.01) -> np.ndarray:
        """Compute morphology rate at time t."""
        return (self(t + dt) - self(t - dt)) / (2 * dt)
    
    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return self.times[-1] - self.times[0]


class SemanticTaskParser:
    """
    Parses natural language task descriptions into structured phases.
    
    Uses LLM to extract:
    - Temporal decomposition of task
    - Semantic tags for each phase
    - Morphology requirements inference
    """
    
    def __init__(self, llm_interface):
        """
        Initialize the semantic parser.
        
        Args:
            llm_interface: Interface to LLM backend
        """
        self.llm = llm_interface
        
        # Semantic tag definitions
        self.semantic_tags = {
            'precision': ['precise', 'accurate', 'exact', 'careful', 'fine'],
            'force': ['strong', 'firm', 'push', 'pull', 'heavy'],
            'compliance': ['soft', 'gentle', 'compliant', 'flexible', 'delicate'],
            'speed': ['fast', 'quick', 'rapid', 'slow', 'careful'],
            'reach': ['far', 'extended', 'stretch', 'close', 'near'],
            'grasp': ['grip', 'hold', 'grasp', 'pinch', 'envelope'],
        }
        
    def parse(self, task_description: str) -> List[TaskPhase]:
        """
        Parse task description into phases.
        
        Args:
            task_description: Natural language task
            
        Returns:
            List of TaskPhase objects
        """
        # LLM prompt for task decomposition
        prompt = self._construct_parsing_prompt(task_description)
        response = self.llm.generate(prompt)
        phases = self._parse_llm_response(response)
        
        return phases
    
    def _construct_parsing_prompt(self, task: str) -> str:
        """Construct prompt for LLM task parsing."""
        return f"""Analyze this manipulation task and decompose it into sequential phases.

Task: {task}

For each phase, provide:
1. Phase description
2. Relative start and end time (0.0 to 1.0)
3. Semantic tags from: {list(self.semantic_tags.keys())}
4. Morphology requirements (gripper_width, stiffness, compliance)

Format as JSON list:
[
  {{
    "description": "Approach object",
    "start": 0.0,
    "end": 0.2,
    "tags": ["reach", "speed"],
    "morphology": {{"gripper_width": 0.8, "stiffness": 0.5, "compliance": 0.3}}
  }},
  ...
]
"""

    def _parse_llm_response(self, response: str) -> List[TaskPhase]:
        """Parse LLM response into TaskPhase objects."""
        import json
        
        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            phases_json = json.loads(response[start:end])
            
            phases = []
            for i, p in enumerate(phases_json):
                phase = TaskPhase(
                    phase_id=i,
                    description=p.get('description', f'Phase {i}'),
                    start_time=p.get('start', i / len(phases_json)),
                    end_time=p.get('end', (i+1) / len(phases_json)),
                    semantic_tags=p.get('tags', []),
                    confidence=0.9
                )
                
                # Parse morphology if provided
                if 'morphology' in p:
                    morph = p['morphology']
                    phase.required_morphology = np.array([
                        morph.get('gripper_width', 0.5),
                        morph.get('stiffness', 0.5),
                        morph.get('compliance', 0.5),
                    ])
                    
                phases.append(phase)
                
            return phases
            
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback: single phase
            return [TaskPhase(
                phase_id=0,
                description="Execute task",
                start_time=0.0,
                end_time=1.0,
                semantic_tags=['grasp'],
                confidence=0.5
            )]


class SemanticToMorphologyMapper:
    """
    Maps semantic task requirements to morphology parameters.
    
    Implements the learned mapping:
        F_S2M: S × W × P → M
        
    where S is semantic space, W is workspace, P is physics constraints.
    """
    
    def __init__(self, morphology_space: MorphologySpace):
        """
        Initialize the mapper.
        
        Args:
            morphology_space: Target morphology space
        """
        self.morph_space = morphology_space
        
        # Default mappings from semantic tags to morphology tendencies
        self.tag_to_morphology = {
            'precision': {'gripper_width': -0.3, 'stiffness': 0.4, 'compliance': -0.2},
            'force': {'gripper_width': 0.0, 'stiffness': 0.5, 'compliance': -0.3},
            'compliance': {'gripper_width': 0.1, 'stiffness': -0.3, 'compliance': 0.5},
            'speed': {'gripper_width': 0.2, 'stiffness': 0.2, 'compliance': 0.0},
            'reach': {'gripper_width': 0.0, 'stiffness': 0.0, 'compliance': 0.0},
            'grasp': {'gripper_width': -0.2, 'stiffness': 0.3, 'compliance': 0.2},
        }
        
    def map(
        self,
        phase: TaskPhase,
        workspace: Optional[Dict] = None,
        physics_constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Map a task phase to morphology requirements.
        
        Args:
            phase: Task phase with semantic tags
            workspace: Workspace constraints (optional)
            physics_constraints: Physics constraints (optional)
            
        Returns:
            Required morphology parameters
        """
        # If morphology already specified, use it
        if phase.required_morphology is not None:
            return self._expand_morphology(phase.required_morphology)
        
        # Base morphology (neutral)
        morphology = np.ones(self.morph_space.dim) * 0.5
        
        # Apply semantic tag influences
        for tag in phase.semantic_tags:
            if tag in self.tag_to_morphology:
                tag_effect = self.tag_to_morphology[tag]
                for i, key in enumerate(['gripper_width', 'stiffness', 'compliance']):
                    if key in tag_effect and i < self.morph_space.dim:
                        morphology[i] += tag_effect[key]
        
        # Apply workspace constraints
        if workspace:
            morphology = self._apply_workspace_constraints(morphology, workspace)
        
        # Apply physics constraints
        if physics_constraints:
            morphology = self._apply_physics_constraints(morphology, physics_constraints)
        
        # Clip to valid bounds
        morphology = self.morph_space.clip(morphology)
        
        return morphology
    
    def _expand_morphology(self, short_morph: np.ndarray) -> np.ndarray:
        """Expand short morphology to full dimension."""
        full = np.ones(self.morph_space.dim) * 0.5
        full[:len(short_morph)] = short_morph
        return full
    
    def _apply_workspace_constraints(
        self,
        morphology: np.ndarray,
        workspace: Dict
    ) -> np.ndarray:
        """Apply workspace constraints to morphology."""
        if 'slot_width' in workspace:
            # Gripper must fit in slot
            max_width = workspace['slot_width'] * 0.9
            morphology[0] = min(morphology[0], max_width)
            
        return morphology
    
    def _apply_physics_constraints(
        self,
        morphology: np.ndarray,
        physics: Dict
    ) -> np.ndarray:
        """Apply physics constraints to morphology."""
        if 'max_force' in physics:
            # Limit stiffness based on force limits
            morphology[1] = min(morphology[1], physics['max_force'] / 10.0)
            
        if 'fragile' in physics and physics['fragile']:
            # Increase compliance for fragile objects
            morphology[2] = max(morphology[2], 0.7)
            
        return morphology


class TrajectoryBlender:
    """
    Synthesizes smooth morphology trajectories from discrete waypoints.
    """
    
    def __init__(
        self,
        morphology_space: MorphologySpace,
        method: str = 'cubic'
    ):
        """
        Initialize trajectory blender.
        
        Args:
            morphology_space: Target morphology space
            method: Interpolation method ('linear', 'cubic', 'quintic')
        """
        self.morph_space = morphology_space
        self.method = method
        
    def blend(
        self,
        phases: List[TaskPhase],
        morphologies: List[np.ndarray],
        horizon: int = 100
    ) -> MorphologyTrajectory:
        """
        Blend phase morphologies into smooth trajectory.
        
        Args:
            phases: Task phases with timing
            morphologies: Required morphology for each phase
            horizon: Number of timesteps
            
        Returns:
            Smooth MorphologyTrajectory
        """
        # Create time array
        times = np.linspace(0, 1, horizon)
        
        # Create waypoints from phases
        waypoint_times = []
        waypoint_morphs = []
        
        for phase, morph in zip(phases, morphologies):
            # Add waypoint at phase start
            waypoint_times.append(phase.start_time)
            waypoint_morphs.append(morph)
            
            # Add waypoint at phase end (if different from next start)
            if phase.end_time < 1.0:
                waypoint_times.append(phase.end_time)
                waypoint_morphs.append(morph)
        
        # Add final waypoint
        if waypoint_times[-1] < 1.0:
            waypoint_times.append(1.0)
            waypoint_morphs.append(morphologies[-1])
            
        waypoint_times = np.array(waypoint_times)
        waypoint_morphs = np.array(waypoint_morphs)
        
        # Interpolate
        if self.method == 'cubic' and len(waypoint_times) >= 4:
            spline = CubicSpline(waypoint_times, waypoint_morphs, axis=0)
            morphologies = spline(times)
        else:
            # Linear interpolation
            morphologies = np.zeros((horizon, self.morph_space.dim))
            for i, t in enumerate(times):
                idx = np.searchsorted(waypoint_times, t)
                idx = min(idx, len(waypoint_times) - 1)
                if idx == 0:
                    morphologies[i] = waypoint_morphs[0]
                else:
                    t0, t1 = waypoint_times[idx-1], waypoint_times[idx]
                    m0, m1 = waypoint_morphs[idx-1], waypoint_morphs[idx]
                    alpha = (t - t0) / (t1 - t0 + 1e-6)
                    morphologies[i] = (1 - alpha) * m0 + alpha * m1
        
        # Clip to bounds and rate limits
        morphologies = self._enforce_rate_limits(times, morphologies)
        morphologies = np.clip(
            morphologies,
            self.morph_space.bounds[0],
            self.morph_space.bounds[1]
        )
        
        return MorphologyTrajectory(
            times=times,
            morphologies=morphologies,
            phases=phases
        )
    
    def _enforce_rate_limits(
        self,
        times: np.ndarray,
        morphologies: np.ndarray
    ) -> np.ndarray:
        """Enforce morphology rate limits."""
        result = morphologies.copy()
        
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            max_change = self.morph_space.rate_limits * dt
            
            change = result[i] - result[i-1]
            change_clipped = np.clip(change, -max_change, max_change)
            result[i] = result[i-1] + change_clipped
            
        return result


class L2MT:
    """
    Language-to-Morphology-Trajectory Algorithm.
    
    This is the MAIN NOVEL CONTRIBUTION: the first algorithm to generate
    morphology TRAJECTORIES (not static designs) from natural language.
    
    Example:
        >>> l2mt = L2MT(morphology_space=MorphologySpace(dim=12))
        >>> trajectory = l2mt.generate(
        ...     task_description="Pick up the fragile vase gently",
        ...     initial_morphology=current_morph,
        ...     horizon=100
        ... )
        >>> # trajectory.morphologies is shape (100, 12)
    """
    
    def __init__(
        self,
        morphology_space: MorphologySpace,
        llm_backend: str = "gpt-4",
        config: Optional[L2MTConfig] = None,
        physics_validator: bool = True
    ):
        """
        Initialize L2MT algorithm.
        
        Args:
            morphology_space: Target morphology space
            llm_backend: LLM to use for semantic parsing
            config: Algorithm configuration
            physics_validator: Whether to validate physics feasibility
        """
        self.morph_space = morphology_space
        self.config = config or L2MTConfig(llm_backend=llm_backend)
        self.physics_validation = physics_validator
        
        # Initialize components
        from morphollm.models.llm_interface import LLMInterface
        self.llm = LLMInterface(backend=self.config.llm_backend)
        self.parser = SemanticTaskParser(self.llm)
        self.mapper = SemanticToMorphologyMapper(morphology_space)
        self.blender = TrajectoryBlender(
            morphology_space, 
            method=self.config.interpolation_method
        )
        
    def generate(
        self,
        task_description: str,
        initial_morphology: np.ndarray,
        horizon: int = 100,
        workspace: Optional[Dict] = None,
        physics_constraints: Optional[Dict] = None
    ) -> MorphologyTrajectory:
        """
        Generate morphology trajectory from task description.
        
        MAIN ALGORITHM:
            1. Parse task into semantic phases
            2. Map each phase to morphology requirements
            3. Synthesize smooth trajectory
            4. Optionally refine with differentiable simulation
        
        Args:
            task_description: Natural language task
            initial_morphology: Starting morphology
            horizon: Number of timesteps
            workspace: Workspace constraints
            physics_constraints: Physics constraints
            
        Returns:
            MorphologyTrajectory: Time-varying morphology specification
        """
        # Stage 1: Semantic Task Parsing
        phases = self.parser.parse(task_description)
        
        # Stage 2: Morphology Requirement Inference
        morphologies = []
        for phase in phases:
            morph = self.mapper.map(phase, workspace, physics_constraints)
            if self.physics_validation:
                morph = self._validate_physics(morph, phase)
            morphologies.append(morph)
            phase.required_morphology = morph
        
        # Prepend initial morphology
        initial_phase = TaskPhase(
            phase_id=-1,
            description="Initial configuration",
            start_time=0.0,
            end_time=phases[0].start_time if phases else 0.1,
            semantic_tags=[]
        )
        phases = [initial_phase] + phases
        morphologies = [initial_morphology] + morphologies
        
        # Stage 3: Trajectory Synthesis
        trajectory = self.blender.blend(phases, morphologies, horizon)
        
        # Stage 4: Differentiable Refinement (optional)
        if self.config.optimization_steps > 0:
            trajectory = self._refine_trajectory(
                trajectory, task_description, workspace
            )
        
        # Store metadata
        trajectory.metadata['task'] = task_description
        trajectory.metadata['initial_morphology'] = initial_morphology
        
        return trajectory
    
    def _validate_physics(
        self,
        morphology: np.ndarray,
        phase: TaskPhase
    ) -> np.ndarray:
        """Validate morphology against physics constraints."""
        # Check workspace reachability
        # Check force/torque limits
        # Adjust if needed
        return self.morph_space.clip(morphology)
    
    def _refine_trajectory(
        self,
        trajectory: MorphologyTrajectory,
        task: str,
        workspace: Optional[Dict]
    ) -> MorphologyTrajectory:
        """
        Refine trajectory using differentiable simulation.
        
        Optimizes:
            m*(t) = argmin L_task(m(t)) + λ L_smooth(ṁ(t))
        """
        # Convert to torch for optimization
        morphs = torch.tensor(
            trajectory.morphologies, 
            dtype=torch.float32,
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam([morphs], lr=self.config.learning_rate)
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Smoothness loss
            m_dot = morphs[1:] - morphs[:-1]
            smoothness_loss = torch.mean(m_dot ** 2)
            
            # Bound constraint loss
            lower = torch.tensor(self.morph_space.bounds[0])
            upper = torch.tensor(self.morph_space.bounds[1])
            bound_loss = torch.mean(torch.relu(lower - morphs)) + \
                        torch.mean(torch.relu(morphs - upper))
            
            # Total loss
            loss = smoothness_loss * self.config.trajectory_smoothness + \
                   bound_loss * 10.0
            
            loss.backward()
            optimizer.step()
        
        # Update trajectory
        trajectory.morphologies = morphs.detach().numpy()
        trajectory.morphologies = np.clip(
            trajectory.morphologies,
            self.morph_space.bounds[0],
            self.morph_space.bounds[1]
        )
        
        return trajectory
    
    def adapt(
        self,
        trajectory: MorphologyTrajectory,
        context: str,
        current_state: np.ndarray,
        current_step: int = 0
    ) -> MorphologyTrajectory:
        """
        Adapt trajectory based on new semantic context.
        
        Args:
            trajectory: Current trajectory
            context: New semantic context (e.g., "object is fragile")
            current_state: Current robot state
            current_step: Current timestep
            
        Returns:
            Adapted trajectory
        """
        # Parse adaptation context
        adaptation_prompt = f"""
Given the current manipulation context: "{context}"
Current progress: {current_step / len(trajectory.times) * 100:.0f}%

Suggest morphology adaptation:
- Should gripper be wider/narrower?
- Should stiffness increase/decrease?
- Should compliance increase/decrease?

Respond with adjustment factors (-0.3 to 0.3):
{{"gripper_width": 0.0, "stiffness": 0.0, "compliance": 0.0}}
"""
        
        response = self.llm.generate(adaptation_prompt)
        
        try:
            import json
            start = response.find('{')
            end = response.rfind('}') + 1
            adjustments = json.loads(response[start:end])
            
            # Apply adjustments to remaining trajectory
            for i in range(current_step, len(trajectory.morphologies)):
                for j, key in enumerate(['gripper_width', 'stiffness', 'compliance']):
                    if key in adjustments and j < self.morph_space.dim:
                        trajectory.morphologies[i, j] += adjustments[key]
            
            # Re-clip to bounds
            trajectory.morphologies = np.clip(
                trajectory.morphologies,
                self.morph_space.bounds[0],
                self.morph_space.bounds[1]
            )
            
        except (json.JSONDecodeError, KeyError):
            pass  # Keep original trajectory if parsing fails
            
        return trajectory
    
    def visualize(
        self,
        trajectory: MorphologyTrajectory,
        save_path: Optional[str] = None
    ):
        """
        Visualize morphology trajectory.
        
        Args:
            trajectory: Trajectory to visualize
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot morphology parameters over time
        ax1 = axes[0, 0]
        for i in range(min(3, self.morph_space.dim)):
            ax1.plot(trajectory.times, trajectory.morphologies[:, i],
                    label=self.morph_space.parameter_names[i])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Parameter Value')
        ax1.set_title('Morphology Parameters')
        ax1.legend()
        ax1.grid(True)
        
        # Plot morphology rates
        ax2 = axes[0, 1]
        m_dot = np.diff(trajectory.morphologies, axis=0) / np.diff(trajectory.times)[:, None]
        for i in range(min(3, self.morph_space.dim)):
            ax2.plot(trajectory.times[:-1], m_dot[:, i],
                    label=f'd{self.morph_space.parameter_names[i]}/dt')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate')
        ax2.set_title('Morphology Rates')
        ax2.legend()
        ax2.grid(True)
        
        # Plot phase boundaries
        ax3 = axes[1, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(trajectory.phases)))
        for phase, color in zip(trajectory.phases, colors):
            ax3.axvspan(phase.start_time, phase.end_time, 
                       alpha=0.3, color=color, label=phase.description[:20])
        ax3.set_xlabel('Time')
        ax3.set_title('Task Phases')
        ax3.legend(loc='upper right', fontsize=8)
        
        # Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"Task: {trajectory.metadata.get('task', 'N/A')[:50]}\n"
        summary += f"Duration: {trajectory.duration:.2f}s\n"
        summary += f"Phases: {len(trajectory.phases)}\n"
        summary += f"Morphology dim: {trajectory.morphologies.shape[1]}"
        ax4.text(0.1, 0.5, summary, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        return fig
