"""
MorphoLLM: Language-Guided Morphological Trajectory Synthesis
for Adaptive Robotic Manipulators

A paradigm-shifting framework that treats morphology as a continuous
control variable during manipulation, guided by LLM semantic reasoning.

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
Institution: EPFL CREATE Lab
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from morphollm.core.extended_config_space import (
    ExtendedConfigSpace,
    MorphologySpace,
    JointSpace,
)
from morphollm.core.morphology_dynamics import MorphologyDynamics
from morphollm.core.stability_analysis import StabilityAnalyzer

from morphollm.algorithms.l2mt import L2MT, L2MTConfig
from morphollm.algorithms.sm_mpc import SMMPC, SMMPCConfig

from morphollm.models.llm_interface import LLMInterface
from morphollm.models.semantic_parser import SemanticParser
from morphollm.models.morphology_encoder import MorphologyEncoder

from morphollm.physics.differentiable_dynamics import DifferentiableDynamics
from morphollm.physics.workspace_analysis import WorkspaceAnalyzer

from morphollm.simulation.manipulator_env import ManipulatorEnv

__all__ = [
    # Core
    "ExtendedConfigSpace",
    "MorphologySpace", 
    "JointSpace",
    "MorphologyDynamics",
    "StabilityAnalyzer",
    # Algorithms
    "L2MT",
    "L2MTConfig",
    "SMMPC",
    "SMMPCConfig",
    # Models
    "LLMInterface",
    "SemanticParser",
    "MorphologyEncoder",
    # Physics
    "DifferentiableDynamics",
    "WorkspaceAnalyzer",
    # Simulation
    "ManipulatorEnv",
]


class MorphoLLMPipeline:
    """
    End-to-end pipeline for language-guided morphological manipulation.
    
    This class provides a unified interface for the complete MorphoLLM
    workflow: task parsing, morphology trajectory generation, and
    closed-loop execution with semantic adaptation.
    
    Example:
        >>> pipeline = MorphoLLMPipeline.from_pretrained("morphollm-base")
        >>> result = pipeline.execute(
        ...     task="Pick up the fragile vase",
        ...     robot=my_robot,
        ...     visualize=True
        ... )
    """
    
    def __init__(
        self,
        l2mt: L2MT,
        sm_mpc: SMMPC,
        llm_interface: LLMInterface,
        config: dict = None
    ):
        """
        Initialize the MorphoLLM pipeline.
        
        Args:
            l2mt: Language-to-Morphology-Trajectory algorithm instance
            sm_mpc: Semantic Morphological MPC controller
            llm_interface: Interface to the LLM backend
            config: Optional configuration dictionary
        """
        self.l2mt = l2mt
        self.sm_mpc = sm_mpc
        self.llm = llm_interface
        self.config = config or {}
        
    @classmethod
    def from_pretrained(cls, model_name: str = "morphollm-base"):
        """
        Load a pretrained MorphoLLM pipeline.
        
        Args:
            model_name: Name of the pretrained model
            
        Returns:
            MorphoLLMPipeline: Initialized pipeline
        """
        from morphollm.utils.model_loader import load_pretrained
        return load_pretrained(model_name)
    
    def execute(
        self,
        task: str,
        robot,
        workspace=None,
        visualize: bool = False,
        max_steps: int = 1000
    ):
        """
        Execute a manipulation task with morphological adaptation.
        
        Args:
            task: Natural language task description
            robot: Robot interface object
            workspace: Optional workspace specification
            visualize: Whether to visualize execution
            max_steps: Maximum number of control steps
            
        Returns:
            ExecutionResult: Result object with success status and metrics
        """
        from morphollm.utils.execution import ExecutionResult
        
        # Generate morphology trajectory from task
        morph_traj = self.l2mt.generate(
            task_description=task,
            initial_morphology=robot.get_morphology(),
            workspace=workspace
        )
        
        # Execute with SM-MPC
        step = 0
        adaptations = 0
        
        while step < max_steps and not robot.task_complete():
            state = robot.get_extended_state()
            perception = robot.get_perception()
            
            # Check for semantic adaptation
            if self._needs_adaptation(perception, state):
                morph_traj = self._adapt_trajectory(
                    morph_traj, perception, state, task
                )
                adaptations += 1
            
            # Compute and apply control
            u_ext = self.sm_mpc.compute(
                state=state,
                reference=morph_traj,
                step=step
            )
            robot.apply_control(u_ext)
            
            if visualize:
                robot.render()
                
            step += 1
        
        return ExecutionResult(
            success=robot.task_complete(),
            steps=step,
            num_adaptations=adaptations,
            final_state=robot.get_extended_state()
        )
    
    def _needs_adaptation(self, perception, state):
        """Check if semantic adaptation is needed."""
        # Use LLM to assess if adaptation is required
        return self.llm.assess_adaptation_need(perception, state)
    
    def _adapt_trajectory(self, trajectory, perception, state, task):
        """Adapt the morphology trajectory based on new information."""
        context = self.llm.analyze_situation(perception, state, task)
        return self.l2mt.adapt(trajectory, context, state)
