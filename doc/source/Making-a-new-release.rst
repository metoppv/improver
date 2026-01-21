Making a new release
====================

New release steps:

1. Inform the core developers across institutions and wait for approval.
2. On the `Code page for improver <https://github.com/metoppv/improver/>` 
   click on the Tags button (with a tag icon and a number of existing Tags)
   to see existing tags.
3. Click on the Releases button (next to Tags) to see existing releases.
4. Click on the Draft a new release button.
5. On the Draft a new release page, click the Select tag dropdown menu,
   and create a new tag by typing in the new version number (e.g., ``1.18.8``).
6. Add a description to the release to describe changes that have been made 
   since the last release, then click the Publish release button.
   In most cases, you can simply use the "Generate release notes" button 
   to fill in the release title and description. The title will be derived
   from the tag name, and the description will be automatically generated
   based on the changes included in that release.
7. At some point after publishing the release, github will create a compressed
   ``.tar.gz`` file of the source code for that release.
   After a release is published, a pull request is typically created automatically
   in the `improver-feedstock repository <https://github.com/conda-forge/improver-feedstock>`_
   to update the conda-forge package. Administrators of that feedstock
   will be notified and will handle reviewing and merging the PR.

Example: https://github.com/metoppv/improver/releases/tag/1.18.6
