#!/usr/bin/env bats

@test "nbhood no arguments" {
  run improver nbhood
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__TEXT__' || true
usage: improver-nbhood [-h]
                       [--radius-in-km RADIUS | --radii-in-km-by-lead-time \
RADIUS_BY_LEAD_TIME RADIUS_BY_LEAD_TIME]
                       INPUT_FILE OUTPUT_FILE
__TEXT__
  [[ "$output" =~ "$expected" ]]
}
