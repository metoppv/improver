# IMPROVER acceptance test data

This repository provides the data required to run the [IMPROVER](https://github.com/metoppv/improver) acceptance tests.

## Using the data

To run the IMPROVER acceptance tests:
- clone this repository to a local location.
- set an environment variable that targets this local repository, e.g. `export IMPROVER_ACC_TEST_DIR=/path/to/test/data`
- ensure you are using a suitable python environment (see [environments here](https://github.com/metoppv/improver/tree/master/envs))
- run the tests from within your IMPROVER repository:
  - specifically using `bin/improver-tests cli`
  - as part of the whole set of tests using `bin/improver-tests`

## Adding test data

To modify existing data, or to add new data, please raise a PR to this repository in association with the IMPROVER PR that requires the change.

Some things to keep in mind:
- try to keep the size of each file you add or modify small (ideally < 50KB)
- include plots of the data in any PR you raise to help reviewers understand what is being added

