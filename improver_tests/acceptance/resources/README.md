# IMPROVER acceptance test data

This directory represents that data required to run the [IMPROVER](https://github.com/metoppv/improver) acceptance tests.

## Using data from an alternative location

Set an environment variable that points to an alternative dataset e.g. `export IMPROVER_ACC_TEST_DIR=/path/to/test/data`

## Adding test data

Any files other than README.md and LICENSE are setup to be tracked by git lfs as binary files.
This is configured in the root `.gitattributes` file.

Some things to keep in mind:
- try to keep the size of each file you add or modify small (ideally < 50KB)
- include plots of the data in any PR you raise to help reviewers understand what is being added
