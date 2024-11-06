Ticket creation and definition of ready
=======================================

Creating new tickets
--------------------

User stories
~~~~~~~~~~~~

When creating new tickets you should follow the template provided:

   As a X I want Y so that Z

   Related issues: #I, #J

   Optional extra information text goes here

   Acceptance criteria:

   * A
   * B
   * C

You should fill in the first sentence, explaining the functionality
required and who will use it. Make sure you link to any related work,
especially prerequisites. You can add more detailed descriptions and
fill in the acceptance criteria to describe what tasks need to be
completed and what testing might be needed.

A good reference for understanding what a user story is is here:
http://www.romanpichler.com/blog/10-tips-writing-good-user-stories/

Labelling issues
----------------

A number of labels are available for helping the categorise the
available issues. Existing labels are:

Type
~~~~

* Administrative: The issue involves general administrative tasks, such as,
  arranging meetings.
* Bug: The functionality is not behaving as expected.
* Documentation: The addition and/or improvement of documentation, either
  within the code, or in terms of writing documents and reports.
* Enhancement: An enhancement to existing functionality.
* Feature: New functionality that does not yet exist in the project.
* Investigative: A more open-ended investigation, into a technical or
  scientific development, which may or may not lead to future work to implement
  the results of the investigation.
* Maintenance: Improvements to the design, structuring and/or behaviour of a
  function. Can be viewed as the removal of minor technical debt to improve the
  supportability and maintainability.
* Optimisation: An optimisation task mostly focussed on improving the
  timeliness of processing.
* Review: A review of the current status of a piece of functionality is
  required, ahead of starting more work.
* Technical Debt: Previous issues have resulted in the accumulation of
  technical debt. These issues do the groundwork to remove the technical debt.

Inactive:
~~~~~~~~~

* Duplicate: A duplicate issue exists, or is already in progress to
  deal with the features described in this issue. Invalid: The basis of
  this issue is invalid, so it is not actually required.
* Invalid: For technical, scientific, administrative or other reasons,
  the premise of this issue is invalid, so can not be completed.
* Won’t fix: For technical, scientific, administrative or other
  reasons, it is not possible to fix the problem described in this
  issue, or we are acknowledging that we won’t fix this issue as it’s
  low priority, and alternative solutions may need to be investigated.
* On Hold: Work started on this issue, however, the priority of this
  issue decreased, and is now On Hold, potentially until a firmer
  requirement can be established.

Problem:
~~~~~~~~

* Blocked: Progress on the issue is blocked. This may require input
  from others to resolve this issue, or progress of this issue is
  blocked until other issues have been resolved.
* Discussion: This issue is in progress but a broader discussion is
  required to continue making progress, in order to better understand
  the requirements, or obtain input into potential solutions to the
  problem.
* Help Wanted: Assistance is required, potentially in terms of an
  informal code review, or a dialogue to help make decisions regarding
  e.g. structuring, style preferences, etc.

Ready to discuss
----------------

Tickets are ready for discussion if as much information as possible has
been added to the ticket by the person raising the issue. We will
discuss the work to ensure that the acceptance criteria are clear and
everyone understands what is involved. The splitting up of the ticket
into reasonable sized chunks also needs to be considered at this stage,
to aid reviewing and to ensure that the simplest and highest priority
functionality is implemented first.

Tickets should:

* Be independent
* Be estimate-able
* Be testable
* Have acceptance criteria.

We also should make an effort to:

* Make it clear how we show the product owner it is done.
* Use the description box to provide more information as appropriate.
* Focus on requirements (necessary outputs, processes etc.) not solutions.
  Put suggested solutions in the comments on the tickets for discussion when
  the ticket is started.
* Use user story format 'As ... I want ... so that ...'.
* Be clear enough so there is a shared understanding of the ticket.
* List all dependencies and related work on the ticket.
