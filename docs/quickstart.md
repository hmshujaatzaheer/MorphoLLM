# Quick Start Guide

Welcome to MorphoLLM! This guide will get you up and running in minutes.

## Installation

### Option 1: From PyPI (Recommended)

```bash
pip install morphollm
```

### Option 2: From Source

```bash
git clone https://github.com/hmshujaatzaheer/MorphoLLM.git
cd MorphoLLM
pip install -e .
```

### Option 3: With All Dependencies

```bash
pip install morphollm[all]
```

## Your First MorphoLLM Program

### Step 1: Import the Library

```python
import numpy as np
from morphollm import L2MT, MorphologySpace, ExtendedConfigSpace
```

### Step 2: Define Your Morphology Space

```python
# Create a 6-dimensional morphology space
morph_space = MorphologySpace(
    dim=6,
    parameter_names=[
        "gripper_width",
        "stiffness", 
        "compliance",
        "wrist_angle",
        "finger_spread",
        "palm_curvature"
    ]
)
```

### Step 3: Initialize L2MT

```python
# Create L2MT with mock backend (no API key needed)
l2mt = L2MT(
    morphology_space=morph_space,
    llm_backend="mock"  # Use "gpt-4" for production
)
```

### Step 4: Generate a Morphology Trajectory

```python
# Define your manipulation task
task = "Pick up a fragile wine glass and place it on the shelf"

# Starting morphology
initial_morph = np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.5])

# Generate trajectory
trajectory = l2mt.generate(
    task_description=task,
    initial_morphology=initial_morph,
    horizon=100
)

print(f"Generated {len(trajectory.times)} timesteps")
print(f"Task phases: {len(trajectory.phases)}")
```

### Step 5: Use the Trajectory

```python
# Sample morphology at different times
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    m = trajectory(t)
    print(f"t={t:.2f}: gripper={m[0]:.2f}, compliance={m[2]:.2f}")
```

## Using SM-MPC Control

For real-time control with morphology adaptation:

```python
from morphollm import SMMPC, SMMPCConfig
from morphollm.core.morphology_dynamics import MorphologyDynamics

# Create configuration space
config_space = ExtendedConfigSpace(joint_dim=7, morphology_dim=6)

# Create dynamics model
dynamics = MorphologyDynamics(config_space)

# Create controller
controller = SMMPC(
    config_space=config_space,
    dynamics=dynamics,
    config=SMMPCConfig(horizon=20)
)

# In your control loop:
# u_ext = controller.compute(state, reference_trajectory, step)
# robot.apply_control(u_ext)
```

## Using Real LLM Backends

### OpenAI GPT-4

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

l2mt = L2MT(
    morphology_space=morph_space,
    llm_backend="gpt-4"
)
```

### Anthropic Claude

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

l2mt = L2MT(
    morphology_space=morph_space,
    llm_backend="claude"
)
```

## Complete Example

See the full examples in the `examples/` directory:

- `basic_l2mt_demo.py` - L2MT trajectory generation
- `sm_mpc_control_demo.py` - SM-MPC closed-loop control

Run them with:

```bash
cd examples
python basic_l2mt_demo.py
python sm_mpc_control_demo.py
```

## Next Steps

1. **Read the API Reference** - Detailed documentation of all classes
2. **Explore Examples** - More complex use cases
3. **Theory Guide** - Mathematical foundations
4. **Contributing** - How to contribute to MorphoLLM

## Getting Help

- **Documentation**: [GitHub README](https://github.com/hmshujaatzaheer/MorphoLLM)
- **Issues**: [GitHub Issues](https://github.com/hmshujaatzaheer/MorphoLLM/issues)
- **Email**: shujabis@gmail.com
