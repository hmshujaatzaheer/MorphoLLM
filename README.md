<div align="center">

# 🤖 MorphoLLM

### Language-to-Morphology Translation for Generative Robotic Manipulator Design

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()

**Bridging Natural Language Understanding and Physical Robot Morphology Generation**

[Overview](#-overview) • [Key Contributions](#-key-contributions) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Citation](#-citation)

</div>

---

## 🔬 Research Context

**The Problem:** Designing robotic manipulators traditionally requires extensive engineering expertise, iterative prototyping, and manual optimization. While Large Language Models (LLMs) have shown remarkable capabilities in reasoning and code generation, they have not been effectively leveraged for *physical robot design*—a domain requiring understanding of kinematics, dynamics, workspace requirements, and manufacturing constraints.

**Our Solution:** MorphoLLM introduces a framework that translates natural language task descriptions into optimized manipulator morphologies. By treating robot design as a language-to-structure translation problem, we enable non-experts to generate functional manipulator designs through intuitive descriptions like *"a robot arm for precise electronic assembly with 50cm reach."*

---

## 🎯 Overview

MorphoLLM addresses the gap between language understanding and physical robot design. Current approaches to manipulator design either require deep engineering expertise or produce generic solutions that don't adapt to specific task requirements. 

This framework introduces:

| Component | Description |
|-----------|-------------|
| **L2MT Module** | Language-to-Morphology Translation using LLM-guided design generation |
| **Physics-Aware Generator** | Ensures generated designs satisfy physical constraints (torque limits, workspace, reachability) |
| **SM-MPC Controller** | Shape-Memory Model Predictive Control for morphology-aware trajectory execution |
| **Continuous Morphology Space** | Differentiable representation enabling gradient-based optimization |

---

## 🌟 Key Contributions

### 1. Language-to-Morphology Translation (L2MT)

Transform natural language specifications into morphological parameters:

```
"A 6-DOF arm for welding with 1.2m reach and high payload"
                    ↓
    [link_lengths, joint_types, actuator_specs, ...]
```

### 2. Physics-Aware Semantic Generator

Ensures generated designs are physically realizable:

- **Kinematic Feasibility:** Workspace coverage, singularity avoidance
- **Dynamic Constraints:** Torque limits, inertia management
- **Manufacturing Viability:** Standard components, assembly constraints

### 3. Morphology as Continuous Control Variable

Unlike discrete design choices, we represent morphology in a continuous, differentiable space:

```
α ∈ ℳ ⊂ ℝⁿ  where  ℳ = {α : g(α) ≥ 0}
```

Enabling gradient-based optimization of physical structure alongside control.

### 4. Shape-Memory MPC (SM-MPC)

Control framework that maintains awareness of morphological state:

```
min  Σₖ ||xₖ - xᵣₑf||²_Q + ||uₖ||²_R + ||Δαₖ||²_S
s.t. xₖ₊₁ = f(xₖ, uₖ, αₖ)
     α ∈ ℳ_feasible
```

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0 or higher
- Git

### Setup (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/hmshujaatzaheer/MorphoLLM.git

# Navigate to directory
cd MorphoLLM

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch (CPU version - for GPU see pytorch.org)
pip install torch torchvision

# Install the package
pip install -e .

# Verify installation
python -c "import morphollm; print(f'MorphoLLM v{morphollm.__version__} installed successfully')"
```

### Quick Install (PowerShell)

```powershell
git clone https://github.com/hmshujaatzaheer/MorphoLLM.git; cd MorphoLLM; pip install torch; pip install -e .
```

---

## 🚀 Quick Start

### Basic Language-to-Morphology Translation

```python
from morphollm import L2MTModule, MorphologyGenerator

# Initialize the L2MT module
l2mt = L2MTModule()

# Natural language task description
task_description = """
Design a robotic manipulator for precision electronics assembly.
Requirements:
- Reach: 50cm workspace radius
- Payload: 500g maximum
- Precision: 0.1mm repeatability
- Environment: Clean room compatible
"""

# Generate morphology from description
morphology = l2mt.translate(task_description)

print(f"Generated {morphology.num_joints}-DOF manipulator")
print(f"Link lengths: {morphology.link_lengths}")
print(f"Joint types: {morphology.joint_types}")
print(f"Estimated workspace: {morphology.workspace_volume:.3f} m³")
```

### Physics-Aware Design Generation

```python
from morphollm import PhysicsAwareGenerator
from morphollm.core import TaskSpecification, PhysicsConstraints

# Define task specification
task_spec = TaskSpecification(
    workspace_radius=0.5,      # meters
    payload_mass=0.5,          # kg
    precision=0.0001,          # meters
    num_dof=6
)

# Define physics constraints
constraints = PhysicsConstraints(
    max_joint_torque=50.0,     # Nm
    max_joint_velocity=3.14,   # rad/s
    gravity=9.81               # m/s²
)

# Generate physically-valid design
generator = PhysicsAwareGenerator(constraints)
design = generator.generate(task_spec)

# Validate design
validation = design.validate()
print(f"Design valid: {validation.is_valid}")
print(f"Workspace coverage: {validation.workspace_coverage:.1%}")
print(f"Max required torque: {validation.max_torque:.2f} Nm")
```

### Shape-Memory MPC Control

```python
import numpy as np
from morphollm import SMMPCController
from morphollm.core import ManipulatorState

# Initialize controller with morphology awareness
controller = SMMPCController(
    prediction_horizon=20,
    control_dt=0.01,
    morphology_weight=0.1
)

# Current state
current_state = ManipulatorState(
    joint_positions=np.zeros(6),
    joint_velocities=np.zeros(6),
    morphology=design.morphology
)

# Target pose
target_pose = np.array([0.3, 0.2, 0.4, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]

# Compute optimal control with morphology awareness
control_action = controller.compute(current_state, target_pose)

print(f"Joint torques: {control_action.torques}")
print(f"Morphology adjustment: {control_action.morphology_delta}")
```

---

## 🏗️ Architecture

```
MorphoLLM/
├── morphollm/
│   ├── core/                      # Core data structures
│   │   ├── morphology_space.py    # Continuous morphology representation
│   │   ├── task_specification.py  # Task description parsing
│   │   ├── kinematics.py          # Forward/inverse kinematics
│   │   └── dynamics.py            # Manipulator dynamics
│   │
│   ├── algorithms/                # Main algorithms
│   │   ├── l2mt.py                # Language-to-Morphology Translation
│   │   ├── physics_generator.py   # Physics-aware design generation
│   │   ├── sm_mpc.py              # Shape-Memory MPC controller
│   │   └── optimization.py        # Morphology optimization
│   │
│   ├── models/                    # Neural network models
│   │   ├── encoder.py             # Task description encoder
│   │   ├── decoder.py             # Morphology parameter decoder
│   │   └── physics_net.py         # Physics constraint network
│   │
│   ├── physics/                   # Physics simulation
│   │   └── constraints.py         # Physical constraint checking
│   │
│   └── utils/                     # Utilities
│       ├── visualization.py       # Design visualization
│       └── validation.py          # Design validation
│
├── examples/                      # Usage demonstrations
│   ├── basic_l2mt_demo.py        # L2MT module demo
│   ├── sm_mpc_control_demo.py    # SM-MPC controller demo
│   └── full_pipeline_demo.py     # Complete pipeline demo
│
├── tests/                         # Unit tests
├── docs/                          # Documentation
└── setup.py                       # Package installation
```

---

## 📊 Research Capabilities

### Supported Manipulator Types

| Type | DOF Range | Applications |
|------|-----------|--------------|
| Serial | 3-7 DOF | Assembly, welding, pick-and-place |
| SCARA | 4 DOF | High-speed assembly |
| Delta | 3-4 DOF | Packaging, sorting |
| Redundant | 7+ DOF | Obstacle avoidance, dexterous manipulation |

### Design Optimization Objectives

- **Workspace Maximization:** Expand reachable volume
- **Manipulability Enhancement:** Improve dexterity metrics
- **Energy Efficiency:** Minimize power consumption
- **Precision Optimization:** Maximize positioning accuracy

---

## 🧪 Running Tests (PowerShell)

```powershell
# Navigate to repository
cd MorphoLLM

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_extended_config_space.py -v

# Run with coverage
python -m pytest tests/ --cov=morphollm --cov-report=html
```

---

## 🔧 Development (PowerShell)

```powershell
# Clone and setup for development
git clone https://github.com/hmshujaatzaheer/MorphoLLM.git
cd MorphoLLM
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch
pip install -e ".[dev]"

# Run linting
pip install flake8
python -m flake8 morphollm --select=E9,F63,F7,F82

# Run examples
python examples/basic_l2mt_demo.py
python examples/sm_mpc_control_demo.py
python examples/full_pipeline_demo.py
```

---

## 📚 Theoretical Foundation

### Morphology Space Representation

```
ℳ = {α ∈ ℝⁿ : g_kin(α) ≥ 0, g_dyn(α) ≥ 0, g_mfg(α) ≥ 0}
```

Where:
- `g_kin(α)` — Kinematic constraints (joint limits, singularities)
- `g_dyn(α)` — Dynamic constraints (torque limits, stability)
- `g_mfg(α)` — Manufacturing constraints (standard sizes, assembly)

### L2MT Formulation

```
α* = argmax_α P(α | T, C)
   = argmax_α P_LLM(α | T) · P_physics(α | C)
```

Where `T` is the task description and `C` are physical constraints.

### SM-MPC Optimization

```
min_{u,Δα} Σₖ₌₀ᴺ [||xₖ - xᵣₑf||²_Q + ||uₖ||²_R + ||Δαₖ||²_S]

s.t.  xₖ₊₁ = f(xₖ, uₖ, αₖ)           (Dynamics)
      αₖ₊₁ = αₖ + Δαₖ                 (Morphology evolution)
      h(xₖ, αₖ) ≤ 0                   (State constraints)
      α ∈ ℳ                           (Feasible morphologies)
```

---

## 🔗 Related Work

This research addresses gaps in existing approaches:

- **Traditional Design:** Manual, expertise-intensive, non-adaptive
- **Topology Optimization:** Numerical, lacks semantic understanding
- **Generative Design:** Often ignores physical realizability
- **LLM Applications:** Previously unexplored for physical robot design

MorphoLLM bridges language understanding with physically-grounded robot design.

---

## 📖 Citation

If you use MorphoLLM in your research, please cite:

```bibtex
@misc{zaheer2026morphollm,
  author       = {Zaheer, H M Shujaat},
  title        = {MorphoLLM: Language-to-Morphology Translation for 
                  Generative Robotic Manipulator Design},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/hmshujaatzaheer/MorphoLLM}},
  note         = {Open-source framework for LLM-guided robot design}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**H M Shujaat Zaheer**

- Research Focus: Generative Robot Design, LLM Applications, AI/ML
- Email: shujabis@gmail.com
- GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```powershell
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/MorphoLLM.git
cd MorphoLLM
git checkout -b feature/your-feature
# Make changes...
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
# Create Pull Request on GitHub
```

---

<div align="center">

**MorphoLLM** — From Language to Robot Morphology

*Democratizing robotic manipulator design through natural language*

⭐ Star this repository if you find it useful for your research!

</div>
