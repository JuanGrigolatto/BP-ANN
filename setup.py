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
    author_email="juan.grigolatto@example.com",
    description="Blood Pressure Estimation from PPG Signals using Artificial Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JuanGrigolatto/BP-ANN",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: Spanish",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical/Dental Instruments and Supplies",
    ],
    python_requires=">=3.9",
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
            "bp-ann-train=src.entrenamiento.train:main",
            "bp-ann-evaluate=src.entrenamiento.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "blood-pressure",
        "ppg",
        "photoplethysmography",
        "neural-networks",
        "deep-learning",
        "pytorch",
        "medical-imaging",
        "signal-processing",
        "meta-learning",
        "few-shot-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/JuanGrigolatto/BP-ANN/issues",
        "Source": "https://github.com/JuanGrigolatto/BP-ANN",
        "Documentation": "https://github.com/JuanGrigolatto/BP-ANN/blob/master/README.md",
    },
)
