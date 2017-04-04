#!/usr/bin/env bats

@test "nbhood no arguments" {
  run improver nbhood
  [[ "$status" -eq 2 ]]
  expected="usage: improver-nbhood [-h] [--radius-in-km RADIUS]\
 INPUT_FILE OUTPUT_FILE"
  [[ "$output" =~ "$expected" ]]
}
