Guidance for reviewing code
===========================

As a Developer:
---------------

1.  Implement the functionality required in the issue, and push the
    branch up to origin.
2.  Check the code with respect to the standards in the Definition of
    Done. Also check that new unit tests do not raise any warnings.
3.  Create a pull request and notify a member of the team so it can be
    assigned to an appropriate reviewer.
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
    column, assign the ticket back to the original developer and notify
    them in preparation for merging into the codebase.

As a Reviewer:
--------------

1. All reviewers are encouraged to add comments to the pull request.

2. Reviewers should:

   a.   **Read the code and post in-line comments for suggested
        alterations.**
   b.   **Ensure unit tests are run and pass.**
   c.   **Ensure command line interface acceptance tests run and pass.
        These must be run on the desktop using bin/improver tests,
        see :doc:`Running-at-your-site` for more information**
   d.   **The Acceptance Criteria defined within the associated issue has
        been met.**
   e.   **The criteria within the Definition of Done has been satisfied.**
   f.   **Ensure their testing is documented on the issue.**

4. Reviewers should post comments to the pull request to show that
   they have completed: a, b, c, d, e, f.

5. Things to consider when reading through the code are:

   a.   **Naming conventions and coding style**

        * Does the code follow the expected protocols for formatting,
          style and naming?
        * Do the names chosen make sense?

   b.   **Design**

        * How does the code fit in with overall IMPROVER design?
        * Is the code in the right place?
        * Is the code logical?
        * Could the new code have reused existing code?

   c.   **Readability and Maintainability**

        * Are the names used meaningful?
        * Are the functions understandable, with the use of the provided
          docstrings and comments?
        * Are any warnings raised when running new unit tests.

   d.   **Functionality**

        * Does the code do what is supposed to?
        * Are errors handled appropriately?
        * Is the code written to be run efficiently?

   e.   **Test coverage**

        * Do the tests provided cover the expected situations?
        * Are the tests understandable?
        * Have edge cases been considered?

6. If this is a first review, the developer should then move the issue
   into 'Second Review' and a second reviewer should ensure that the
   criteria listed above have been met. The Scrum Master can be
   consulted, if necessary.

7. If this is a second review, the developer should assign the issue back
   to the developer and contact them prior to moving the issue into the
   'Done' column. The Scrum Master can be consulted during the Stand-up.
