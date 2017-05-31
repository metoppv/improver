#!/usr/bin/env bats

@test "nbhood --radius-in-km=20 input output" {
  TEST_DIR=$(mktemp -d)
  if [[ -z "${IMPROVER_ACC_TEST_DIR:-}" ]]; then
    skip "Acceptance test directory not defined"
  fi
  if ! which nccmp 1>/dev/null 2>&1; then
    skip "nccmp not installed"
  fi
  # Run neighbourhood processing and check it passes.
  run improver nbhood --radius-in-km=20 \
      "$IMPROVER_ACC_TEST_DIR/nbhood/basic/input.nc" "$TEST_DIR/output.nc"
  [[ "$status" -eq 0 ]]

  # Run nccmp to compare the output and kgo.
  run nccmp -dmNs "$TEST_DIR/output.nc" \
      "$IMPROVER_ACC_TEST_DIR/nbhood/basic/kgo.nc"
  [[ "$status" -eq 0 ]]
  [[ "$output" =~ "are identical." ]]
  rm "$TEST_DIR/output.nc"
  rmdir "$TEST_DIR"
}
