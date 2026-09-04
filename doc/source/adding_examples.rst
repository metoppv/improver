.. _adding_examples:

Adding Examples to the Gallery
===============================

This guide explains how to add new examples to the IMPROVER examples gallery.

Adding a New Example
---------------------

To add a new example to this gallery:

1. Create a Python file in the ``doc/source/examples/`` directory with a 
   descriptive name.

2. Follow the format of existing examples like ``thresholding_simple_example.py``:
   
   - Start with a docstring (triple quotes) containing a title with equals signs 
     above and below
   - Add a description of what the example demonstrates
   - Include author and license information
   - Use ``# %%`` to separate code cells
   - Add comments or section headers (using ``# %%`` followed by description lines) 
     to explain each step

3. Alternatively, you can create a Jupyter notebook and convert it to a Sphinx-Gallery 
   compatible Python file using ``jupytext``::

     jupytext --to py:percent --opt notebook_metadata_filter='-all' your_notebook.ipynb

4. Sphinx-Gallery will automatically:
   
   - Generate a thumbnail from the first figure
   - Convert the script to a reStructuredText page
   - Add it to this gallery
   - Create a downloadable Jupyter notebook version

5. Build the documentation to see your example in the gallery.

Using Real Data in Examples
----------------------------

If you want to create examples that use real data files:

1. Add the data files to the `improver_example_data repository <https://github.com/metoppv/improver_example_data>`_.

2. Create a branch in the ``improver_example_data`` repository with your data files.

3. When creating a pull request in this repository (``improver``), note the PR number 
   (e.g., ``123``). Create a branch in ``improver_example_data`` with the **same number** 
   as the PR number (e.g., branch named ``123``). This ensures that the documentation 
   build for the pull request preview can access the correct version of the example data.
   
   Alternatively, for local development, you can create matching branch names in both 
   repositories instead of using PR numbers.

4. In your example script, use the ``example_data_path()`` function to reference the 
   data files::

     from improver import example_data_path
     
     data_file = example_data_path("subdirectory", "filename.nc")

5. When both pull requests are merged, the documentation will automatically use the 
   example data from the ``master`` branch of ``improver_example_data``.

Managing Dependencies for Examples
-----------------------------------

Examples require appropriate Python packages to be available in multiple environments. 
If your example requires additional modules that are not part of the standard IMPROVER 
dependencies:

1. **For ReadTheDocs builds:** Check if the required module is listed in 
   ``doc/rtd_environment.yml``. If missing, add it to the ``dependencies:`` section. 
   This file specifies the conda environment used when building the documentation on 
   ReadTheDocs.

2. **For GitHub Actions CI checks:** Examples are also run as part of the continuous 
   integration checks using the test environments defined in the ``envs/`` directory 
   (e.g., ``envs/environment_a.yml``, ``envs/environment_b.yml``). Unless the required 
   module is available in **all** of these test environments, some GitHub Actions 
   checks will fail.

3. If a required module is not available in all environments and cannot be added to 
   them, you can make your example skip gracefully by checking for the module's 
   availability at the start of the script::

     try:
         import required_module
     except ImportError:
         import sys
         sys.exit(0)  # Exit cleanly for sphinx-gallery

   This allows sphinx-gallery to skip the example without failing the documentation 
   build or GitHub Actions checks.
