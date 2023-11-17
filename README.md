# IMPROVER

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/metoppv/improver/actions/workflows/scheduled.yml/badge.svg)](https://github.com/metoppv/improver/actions/workflows/scheduled.yml)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f7dcb46e8e1b4110b3d194dba03fe526)](https://www.codacy.com/app/metoppv_tech/improver?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=metoppv/improver&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/f7dcb46e8e1b4110b3d194dba03fe526)](https://www.codacy.com/app/metoppv_tech/improver?utm_source=github.com&utm_medium=referral&utm_content=metoppv/improver&utm_campaign=Badge_Coverage)
[![codecov](https://codecov.io/gh/metoppv/improver/branch/master/graph/badge.svg)](https://codecov.io/gh/metoppv/improver)
[![Documentation Status](https://readthedocs.org/projects/improver/badge/?version=latest)](http://improver.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
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
