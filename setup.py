from setuptools import setup, find_packages

setup(
    name="mcts-routing",
    version="0.1.0",
    description="Diffusion-guided MCTS routing system for FPGA",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
        ],
    },
)

