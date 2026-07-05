"""Installation script for the 'tac_manip' python package."""

import os

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "debugpy",  # for debugging scripts that are run from the terminal, e.g. RL training scripts
    "psutil",
    "nvidia-ml-py",
    "pre-commit",
]

# GPU Taxim imports torch_scatter at runtime. Its wheel must match the active
# Python, PyTorch, and CUDA versions, so pinning one cp311/torch-2.8/cu128 wheel
# here made editable installation fail in the public Isaac Sim 4.5 environment.
# Install the compatible wheel as part of the simulator environment instead;
# the released requirements.txt records the tested torch-2.7/cu128 build.

# Installation operation
setup(
    name="tac_manip",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache 2.0",
    include_package_data=True,
    python_requires=">=3.10,<3.11",
    classifiers=[
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
