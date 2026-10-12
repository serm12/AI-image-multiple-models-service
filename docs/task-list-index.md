# Task list index

Task folders and their generation JSON/images remain authoritative. The list
uses `tasks/.task-list-index.sqlite3` (SQLite WAL) for filter fields, UTC ordering,
counts, pagination, and per-IP sequence numbers. Only tasks on the selected page
are hydrated from their folders, keeping generation status and image URLs fresh.

Indexes cover chronological ordering, favorites and IP grouping. Existing
case-insensitive substring searches keep their semantics and scan database
columns instead of opening every JSON file. User text is bound as SQL parameters.

Parameter writers, storefront events and favorite changes update the affected
index row after the JSON file is closed. Cleanup removes the index row after
removing a task folder. Failed index updates do not fail image generation.

A startup/60-second recovery pass compares file size and nanosecond mtime for
params, events and admin metadata; it only rereads changed files. This recovers
external edits/deletes and missed writes. It is not part of warm list requests.
Initial backfill also occurs on the first query if startup recovery has not run.
Incomplete JSON is retried later; existing index rows are retained while a file
is being rewritten. Files are never modified by backfill.

The index can be rebuilt with `app.services.task_list_index.reconcile()` after
restoring task folders. Keep the SQLite database and its WAL/SHM files on the
same persistent task volume; use SQLite's backup API for live backups, or stop
writers before copying. If the index is unavailable, list requests log the error
and temporarily use the original file scan. No extra Python dependency is needed.

Verification: `python -m unittest tests.test_task_list_index tests.test_admin_tasks
 tests.test_time_settings tests.test_storefront_events tests.test_i18n`.
