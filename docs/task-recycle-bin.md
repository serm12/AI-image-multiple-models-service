# Task recycle bin

The task table supports per-row selection and select-current-page (maximum 100).
`Delete selected` opens a confirmation listing exact IDs. Confirmation moves the
records into the recycle bin; it never immediately deletes image files. Switching
filters/pages clears selection. Success updates the table without a document reload.

The recycle-bin view supports the same filters and pagination, shows each cleanup
date, and allows selected tasks to be restored. Restore preserves task IDs, images,
stars, conversion events and order links. Repeated deletion does not extend the
original retention period. Processing tasks are rejected with a per-task reason.

Markers in `TASKS_DIR/.recycle-bin/<task_id>.json` are durable and separate from
both task files and the rebuildable SQLite list index. They record UTC trash time,
the 30-day expiry, and (when applicable) cleanup start. Both the indexed list and
filesystem fallback distinguish normal/recycled tasks. Rebuilding the index does
not restore discarded tasks.

The application starts an expiry worker on boot and checks hourly. It purges only
expired tasks; delays or downtime never shorten retention. Cleanup deletes all
objects under the exact R2 task prefix, local task files (including thumbnails),
customer-sync jobs, and local cached task/order associations. Actual Shopify orders
and customers are not deleted. Source tasks used by edits are not recursively
followed. CDN/browser copies previously cached outside origin storage can remain
until their cache expires; deleting an R2 object does not invalidate those copies.

Cloud/storage errors preserve the external marker and retry next hour. Once cleanup
has started, restore is disabled because some files may already be gone. Cloud
errors prevent local removal. Markers survive interrupted/partial local deletion
so it can be retried after restart. All admin actions require authenticated access,
same-origin requests, bounded valid task IDs and explicit confirmation. No endpoint
for arbitrary filesystem deletion or immediate permanent deletion is exposed.

Validation:
- `python -m unittest tests.test_task_recycle_bin tests.test_task_list_index tests.test_admin_frontend tests.test_admin_tasks -q`
- `npm run admin:build`
- Browser smoke checks: select/indeterminate, cancel without request, error retry,
  successful inline removal, recycle-bin restore and narrow-screen dialog.
