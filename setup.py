"""
MorphoLLM: Language-Guided Morphological Trajectory Synthesis

Setup script for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="morphollm",
    version="0.1.0",
    author="H M Shujaat Zaheer",
    author_email="shujabis@gmail.com",
    description="Language-Guided Morphological Trajectory Synthesis for Adaptive Robotic Manipulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmshujaatzaheer/MorphoLLM",
    project_urls={
        "Bug Tracker": "https://github.com/hmshujaatzaheer/MorphoLLM/issues",
        "Documentation": "https://github.com/hmshujaatzaheer/MorphoLLM#readme",
        "Source Code": "https://github.com/hmshujaatzaheer/MorphoLLM",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "simulation": [
            "mujoco>=3.0.0",
            "gymnasium>=0.29.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "mujoco>=3.0.0",
            "gymnasium>=0.29.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "morphollm=morphollm.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "robotics",
        "manipulation",
        "llm",
        "morphology",
        "control",
        "mpc",
        "trajectory",
        "deep-learning",
    ],
)
