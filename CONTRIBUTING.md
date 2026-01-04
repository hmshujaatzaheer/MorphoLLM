# Contributing to MorphoLLM

Thank you for your interest in contributing to MorphoLLM! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/MorphoLLM.git
cd MorphoLLM
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e ".[dev]"
```

4. **Run tests to verify setup**
```bash
pytest tests/ -v
```

## 📋 Development Workflow

### Branch Naming Convention

- `feature/` - New features (e.g., `feature/adaptive-mpc`)
- `bugfix/` - Bug fixes (e.g., `bugfix/trajectory-interpolation`)
- `docs/` - Documentation updates (e.g., `docs/api-reference`)
- `refactor/` - Code refactoring (e.g., `refactor/dynamics-module`)

### Making Changes

1. Create a new branch from `main`
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards

3. Write or update tests as needed

4. Run the test suite
```bash
pytest tests/ -v --cov=morphollm
```

5. Format your code
```bash
black morphollm/
isort morphollm/
```

6. Run type checking
```bash
mypy morphollm/
```

7. Commit your changes with a clear message
```bash
git commit -m "feat: add adaptive morphology rate limiting"
```

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: implement semantic morphology adaptation
fix: correct stability bound calculation
docs: update L2MT algorithm documentation
test: add integration tests for SM-MPC
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Mirror the source structure (e.g., `morphollm/core/` → `tests/test_core/`)
- Use descriptive test names
- Include docstrings explaining what each test verifies

### Test Categories

```python
# Unit tests - test individual functions
def test_morphology_space_clip():
    """Test that morphology clipping respects bounds."""
    ...

# Integration tests - test component interactions
def test_l2mt_pipeline_integration():
    """Test full L2MT pipeline from task to trajectory."""
    ...

# Property-based tests (using hypothesis)
@given(arrays(dtype=float, shape=(6,)))
def test_morphology_bounds_invariant(m):
    """Clipped morphology always within bounds."""
    ...
```

### Running Specific Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_extended_config_space.py -v

# Run tests matching pattern
pytest tests/ -k "morphology" -v

# Run with coverage
pytest tests/ --cov=morphollm --cov-report=html
```

## 📖 Documentation Guidelines

### Docstring Format

We use Google-style docstrings:

```python
def compute_morphology_trajectory(
    task: str,
    initial_morphology: np.ndarray,
    horizon: int = 100
) -> MorphologyTrajectory:
    """
    Generate morphology trajectory from task description.
    
    This function implements the L2MT algorithm to convert natural
    language task specifications into time-varying morphology parameters.
    
    Args:
        task: Natural language task description.
        initial_morphology: Starting morphology configuration.
        horizon: Number of timesteps in trajectory.
        
    Returns:
        MorphologyTrajectory containing time-indexed morphology values.
        
    Raises:
        ValueError: If initial_morphology has wrong dimension.
        
    Example:
        >>> trajectory = compute_morphology_trajectory(
        ...     "Pick up fragile vase",
        ...     np.zeros(6),
        ...     horizon=50
        ... )
    """
```

### Updating Documentation

1. Update docstrings when modifying functions
2. Update README.md for user-facing changes
3. Add examples for new features
4. Update API reference in `docs/`

## 🔍 Code Review Process

1. **Submit a Pull Request** with a clear description
2. **Automated checks** will run (tests, linting, type checking)
3. **Maintainer review** - expect feedback within 48 hours
4. **Address feedback** and update the PR
5. **Merge** once approved

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)

## Related Issues
Closes #123
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional LLM backend support (Gemini, local models)
- [ ] MuJoCo simulation environment integration
- [ ] Real robot hardware interfaces
- [ ] Benchmark suite expansion

### Good First Issues
- Documentation improvements
- Example notebooks
- Test coverage expansion
- Code comment improvements

### Research Contributions
- Novel semantic adaptation strategies
- Stability analysis extensions
- Multi-robot morphology coordination

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug Reports**: Open a GitHub Issue with reproduction steps
- **Feature Requests**: Open a GitHub Issue with use case description

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MorphoLLM! 🤖
