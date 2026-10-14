// Shopify launch and authentication parameters are not task filters.
const filterKeys = new Set([
  "task_id", "user", "provider", "client_ip", "time_range", "starred",
  "conversion_status", "page", "page_size", "recycle_bin",
]);
export function taskQuery(search: string): string {
  const clean = new URLSearchParams();
  for (const [key, value] of new URLSearchParams(search)) {
    if (filterKeys.has(key) && value) clean.append(key, value);
  }
  const query = clean.toString();
  return query ? `?${query}` : "";
}
