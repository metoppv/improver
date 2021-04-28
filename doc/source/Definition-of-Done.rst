Definition of done
==================

1) Meets acceptance criteria in issue (if it is a bugfix PR, the bugfix
   PR should accurately spell out the problem being fixed with a user
   story).
2) Meets :doc:`Code-Style-Guide`.
3) All new functionality unit tested if it is reasonable to do so
   (i.e. if the code is anything other than obvious).
4) CLI acceptance tests have been added if CLI has been added. The known
   good output (KGO) for these tests should be added and committed to
   the version controlled acceptance test repository (if accessible).
   Output should usually be 32 bit precision.
5) All tests pass (should be run locally using /bin/improver-tests as
   well as checking all GitHub Actions tests pass).
6) Codacy GitHub check passes (if Codacy raises something unreasonable,
   ask for a tweak to be made to Codacy settings to allow it to pass)
7) Known bugs fixed, or follow-on issues have been raised. Make sure the
   product owner is aware of any follow on tickets that have been
   raised.
8) Reviewed by at least one person, but always at least 2 for serious
   non-bugfix changes.
9) Ensure licence information referred to in the :doc:`Code-Style-Guide`
   is included within any new files.
