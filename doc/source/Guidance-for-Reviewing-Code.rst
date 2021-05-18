Guidance for reviewing code
===========================

As a Developer:
---------------

1.  Implement the functionality required in the issue, and push the
    branch up to origin.
2.  Check the code with respect to the standards in the Definition of
    Done. Also check that new unit tests do not raise any warnings.
3.  Create a pull request. A main reviewer must be assigned, but adding
    multiple reviewers to the pull request is encouraged to help share
    expertise. Please ask for help assigning reviewers if you need it.
4.  The relevant issue should be referenced within the pull request.
5.  Respond to the reviewers’ comments and update the pull request
    accordingly. This will be an iterative process.
6.  (Internal staff only:) Once the first reviewer is satisfied, move
    the ticket forward into the 'Second Review' column.
7.  Ensure that someone is aware of their responsibility to provide a
    second review.
8.  Respond to the second reviewers’ comments and update the pull
    request accordingly. This will be an iterative process.
9.  By the end of the second review, the technical and scientific
    changes, if required, made in the pull request should have been
    thoroughly reviewed. If required from a scientific or technical
    perspective, ensure that a subject matter expert is satisfied with
    the changes.
10. (Internal staff only:) Prior to moving the issue into the 'Done'
    column, notify the Product Owner that the issue has been completed,
    and discuss the functionality implemented.

As a Reviewer:
--------------

1. Issues should be assigned to a main reviewer. Adding multiple
   reviewers to the pull request is encouraged to help share expertise.
   Please ask for help assigning reviewers if you need it.

2. All reviewers are encouraged to add comments to the pull request.

3. Main reviewers should:

  i.   **Read the code and post in-line comments for suggested
       alterations.**
  ii.  **Ensure unit tests are run and pass.**
  iii. **Ensure command line interface acceptance tests run and pass.
       These must be ran on the desktop using bin/improver tests as they
       are not run by travis**
  iv.  **The Acceptance Criteria defined within the associated issue has
       been met.**
  v.   **The criteria within the Definition of Done has been satisfied.**
  vi.  **Ensure their testing is documented on the issue.**

4. Main reviewers should post comments to the pull request to show that
   they have completed: i, ii, iii, iv, v, vi.

5. Things to consider when reading through the code are:

  i.   **Naming conventions and coding style**

    - Does the code follow the expected protocols for formatting,
      style and naming?
    - Do the names chosen make sense?

  ii.  **Design**

    - How does the code fit in with overall IMPROVER design?
    - Is the code in the right place?
    - Is the code logical?
    - Could the new code have reused existing code?

  iii. **Readability and Maintainability**

    - Are the names used meaningful?
    - Are the functions understandable, with the use of the provided
      docstrings and comments?
    - Are any warnings raised when running new unit tests.

  iv.  **Functionality**

    - Does the code do what is supposed to?
    - Are errors handled appropriately?

  v.   **Test coverage**

    - Do the tests provided cover the expected situations?
    - Are the tests understandable?
    - Have edge cases been considered?

6. If this is a first review, the developer should then move the issue
   into 'Second Review' and a second reviewer should ensure that the
   criteria listed above have been met. The Scrum Master can be
   consulted, if necessary.

7. If this is a second review, the developer should contact the Product
   Owner prior to moving the issue into the 'Done' column. The Scrum
   Master can be consulted, if necessary.

Before merging:
---------------

Always rerun Travis before merging a PR. Do this by:

1. At the bottom of the PR, click ``show all checks`` and then ``details`` next
   to the Travis check.
2. Click ``Restart build``

If you haven’t already signed into Travis with GitHub it will prompt you
to do this the first time you try this, and it may take a few minutes to
sync before you can see the page with the option to restart the build.
