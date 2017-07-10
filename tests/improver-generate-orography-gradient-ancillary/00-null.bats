#!/usr/bin/env bats

@test "orography gradient generation no arguments" {
  run improver generate-orography-gradient
  [[ "$status" -eq 2 ]]
  expected="usage: improver-generate-orography-gradient [-h] [--force]
                                            INPUT_FILE_STANDARD OUTPUT_FILE"
  [[ "$output" =~ "$expected" ]]
}
