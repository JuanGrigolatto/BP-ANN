#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup configuration for BP-ANN package
Blood Pressure Estimation using Artificial Neural Networks
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="BP-ANN",
    version="1.0.0",
    author="Juan Grigolatto",
    author_email="juan.grigolatto@uner.edu.ar",
    description="Blood Pressure Estimation using Convolutional Neural Networks and Meta-Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JuanGrigolatto/BP-ANN",
    # find_packages() discovers src/ and its subpackages correctly,
    # consistent with the 'from src.models...' import style used throughout the codebase.
    packages=find_packages(),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bp-ann-train=src.entrenamiento.Entrenamiento:main",
            "bp-ann-train-patientwise=src.entrenamiento.Entrenamiento_patient_subject:main",
            "bp-ann-metatrain=metalearning.Metaentrenamiento:main",
            "bp-ann-fewshot=metalearning.Fewshot:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "blood-pressure",
        "ppg",
        "photoplethysmography",
        "ecg",
        "neural-networks",
        "deep-learning",
        "pytorch",
        "signal-processing",
        "meta-learning",
        "maml",
        "few-shot-learning",
        "bioengineering",
        "non-invasive",
    ],
    project_urls={
        "Bug Reports": "https://github.com/JuanGrigolatto/BP-ANN/issues",
        "Source": "https://github.com/JuanGrigolatto/BP-ANN",
        "Documentation": "https://github.com/JuanGrigolatto/BP-ANN/blob/master/README.md",
    },
)