# IMPROVER

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/metoppv/improver/actions/workflows/scheduled.yml/badge.svg)](https://github.com/metoppv/improver/actions/workflows/scheduled.yml)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f7dcb46e8e1b4110b3d194dba03fe526)](https://www.codacy.com/app/metoppv_tech/improver?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=metoppv/improver&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/f7dcb46e8e1b4110b3d194dba03fe526)](https://www.codacy.com/app/metoppv_tech/improver?utm_source=github.com&utm_medium=referral&utm_content=metoppv/improver&utm_campaign=Badge_Coverage)
[![codecov](https://codecov.io/gh/metoppv/improver/branch/master/graph/badge.svg)](https://codecov.io/gh/metoppv/improver)
[![Documentation Status](https://readthedocs.org/projects/improver/badge/?version=latest)](http://improver.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![DOI](https://zenodo.org/badge/85334761.svg)](https://zenodo.org/badge/latestdoi/85334761)

IMPROVER is a library of algorithms for meteorological post-processing and verification.

## Cloning improver

Clone the Repository:
```
git clone git@github.com:metoppv/improver.git
```

Enter the Cloned Repository Directory:
```
cd improver
```

Fetch LFS data:
```
git lfs fetch
```

After following these steps, you'll have cloned the repository and ensured that the Git LFS data is available locally. Now you can work with the repository, and Git LFS will automatically handle LFS data during operations like checkout, switch branches, etc. If you encounter any issues with LFS data not being updated automatically, you can always manually fetch the LFS data using git lfs fetch.

You can query the status of LFS files locally any time with:
```
git lfs ls-files
```

Note that the `.gitattributes` file configures the git LFS.  It is setup such that all files recursively under `improver_tests/acceptance/resources` are managed by git LFS.

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
mamba create -c conda-forge python=3.7 improver -n improver
```
 
deactivate your mamba environment
```
conda deactivate
```
 
activate your new improver environment
```
conda activate improver
```
