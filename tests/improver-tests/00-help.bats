#!/usr/bin/env bats

@test "tests -h" {
  run improver tests -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
improver tests [--debug] [SUBTEST] 

Run pep8, pylint, documentation, unit and CLI acceptance tests.

Optional arguments:
    --debug         Run in verbose mode (may take longer for CLI)
    -h, --help          Show this message and exit

Arguments:
    SUBTEST         Name of a subtest to run without running the rest.
                    Valid names are: pep8, pylint, pylintE, unit, cli.
                    pep8, pylintE, unit, and cli are the default tests.
__HELP__
  [[ "$output" == "$expected" ]]
}
