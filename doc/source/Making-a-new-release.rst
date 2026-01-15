Making a new release
====================

New release steps:

1. Inform the core developers across institutions and wait for approval.
2. On the command line, check out the commit on master that you want
   to tag. For a given version such as 1.1.0, run:
   `git tag -a 1.1.0 -m "IMPROVER release 1.1.0"`. Then run:
   `git push upstream 1.1.0`.
3. Go to `Draft a new
   release <https://github.com/metoppv/improver/releases/new>`_ page.
   Select your new tag under 'tag version'.
   The **release title** should be the version number (e.g., ``1.1.0``).
   Publish the release after adding any description text.
4. Update the version number and sha256 checksum in the ``meta.yaml``
   file of the conda-forge recipe by opening a pull request in the
   `improver-feedstock <https://github.com/conda-forge/improver-feedstock>`_
   repository. A pull request may be opened automatically for you, in which
   case just check it. The checksum of the compressed ``.tar.gz`` IMPROVER
   source code can be obtained via ``openssl sha256 <file name>``.
   Currently the people with write access to the improver-feedstock
   repository are @benfitzpatrick, @PaulAbernethy, @tjtg, @cpelley and
   @dementipl.
   You can ping one of these people to merge your pull request.
