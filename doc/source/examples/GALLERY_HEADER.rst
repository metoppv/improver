.. _gallery_examples:

Examples
========

This is a gallery of examples to demonstrate the possible usages for IMPROVER.

Adding a New Example
---------------------

To add a new example to this gallery:

1. Create a Python file in the ``doc/source/examples/`` directory with a 
descriptive name.

2. Follow the format of existing examples like ``thresholding_simple_example.py``:
   
   - Start with a docstring (triple quotes) containing a titl.. _gallery_examples:e with equals signs 
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



