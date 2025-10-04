"""
Setup configuration for TSP Heuristics package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tsp-heuristics",
    version="1.0.0",
    author="Lee Sang-cheol",
    author_email="lee.sangcheol@example.com",
    description="Heuristic algorithms for solving the Traveling Salesman Problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/tsp-heuristics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=3.0.0",
            "black>=21.9b0",
            "flake8>=4.0.1",
            "pylint>=2.11.1",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.4.1",
            "notebook>=6.4.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "tsp-solve=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/usuario/tsp-heuristics/issues",
        "Source": "https://github.com/usuario/tsp-heuristics",
        "Documentation": "https://tsp-heuristics.readthedocs.io/",
    },
)