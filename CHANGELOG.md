# Changelog

All notable changes to MorphoLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- MuJoCo simulation environment integration
- Real robot hardware interfaces (Franka, UR5)
- Additional LLM backend support (Gemini, LLaMA)
- Comprehensive benchmark suite

## [0.1.0] - 2026-01-04

### Added
- **Extended Configuration Space** (`ExtendedConfigSpace`)
  - Unified joint and morphology space representation
  - State composition and decomposition utilities
  - Configurable bounds and rate limits

- **Morphology-Augmented Dynamics** (`MorphologyDynamics`)
  - Novel morphology-joint coupling term (J_m)
  - Forward and inverse dynamics computation
  - PyTorch-differentiable implementation

- **Stability Analysis** (`StabilityAnalyzer`)
  - Morphological Stability Theorem implementation
  - Lyapunov-based stability bounds
  - Safe morphology rate computation

- **L2MT Algorithm** (`L2MT`)
  - Semantic task parsing via LLM
  - Morphology requirement inference
  - Smooth trajectory synthesis
  - Differentiable trajectory refinement

- **SM-MPC Controller** (`SMMPC`)
  - Model predictive control with morphology
  - Online semantic adaptation
  - Stability-constrained optimization
  - Warm-starting for real-time performance

- **LLM Interface** (`LLMInterface`)
  - OpenAI GPT-4 support
  - Anthropic Claude support
  - Mock backend for testing

- **Examples**
  - Basic L2MT demonstration
  - SM-MPC control demonstration

- **Testing**
  - Unit tests for core components
  - Integration test framework

### Technical Details
- Python 3.9+ support
- PyTorch 2.0+ for differentiable operations
- NumPy/SciPy for numerical computations

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-04 | Initial release with L2MT and SM-MPC |

[Unreleased]: https://github.com/hmshujaatzaheer/MorphoLLM/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hmshujaatzaheer/MorphoLLM/releases/tag/v0.1.0
