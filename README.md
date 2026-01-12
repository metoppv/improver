# IMPROVER

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/metoppv/improver/actions/workflows/master_update.yml/badge.svg)](https://github.com/metoppv/improver/actions/workflows/master_update.yml)
[![Documentation Status](https://readthedocs.org/projects/improver/badge/?version=latest)](http://improver.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![DOI](https://zenodo.org/badge/85334761.svg)](https://zenodo.org/badge/latestdoi/85334761)

IMPROVER is a library of algorithms for meteorological post-processing and verification.

## Installing improver

### Conda installation

Here we demonstrate the installation of improver via conda with aid of the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager to speedup the process.

install a mamba environment
```
conda create -c conda-forge --override-channels mamba -n mamba
```

activate this mamba environment
```
conda activate mamba
```

install the improver environment using mamba
```
mamba create -c conda-forge python=3 improver -n improver
```

deactivate your mamba environment
```
conda deactivate
```

activate your new improver environment
```
conda activate improver
```

## Pre-commit Hook
OPTIONAL: A pre-commit hook can be added to facilitate the development of this code base.
Ensure that you have python available on the path, then install the pre-commit hook by running `pre-commit install` from within your working copy.
pre-commit checks will run against modified files when you commit from then on.

These pre-commit hooks will run as part of continuous integration to maintain code quality standards in the project.
