.. _improver_examples:

Examples
========

This is a gallery of examples to demonstrate the possible usages for IMPROVER.

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

