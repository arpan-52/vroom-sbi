#!/usr/bin/env python3
"""
Setup script for vroom-sbi.

Install with:
    pip install -e .
    
Then use:
    vroom-sbi train --config config.yaml
    vroom-sbi validate --config config.yaml
    vroom-sbi infer --q "..." --u "..."
"""

from setuptools import setup, find_packages

setup(
    name="vroom-sbi",
    version="1.0.0",
    description="Simulation-Based Inference for RM Synthesis",
    author="VROOM Team",
    python_requires=">=3.8",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.20",
        "torch>=1.10",
        "sbi>=0.18",
        "astropy>=5.0",
        "matplotlib>=3.5",
        "corner>=2.2",
        "tqdm>=4.60",
        "pyyaml>=6.0",
        "huggingface_hub>=0.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "vroom-sbi=src.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
