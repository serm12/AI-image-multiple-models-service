// No DOM, cookies, local storage, admin credentials, or production console logs.
export const ENDPOINT = 'https://image-api.shopupup.com/storefront-events';
const CHECKOUT_EVENTS = new Set(['checkout_started', 'checkout_completed']);

export function checkoutPayloads(eventType, event, shopDomain) {
  const checkout = event?.data?.checkout;
  if (!CHECKOUT_EVENTS.has(eventType) || !checkout || !event.id) return [];
  const tasks = new Map();
  for (const line of checkout.lineItems || []) {
    const properties = Object.fromEntries((line.properties || []).map(p => [p.key, p.value]));
    const taskId = String(properties['_AI Task ID'] || '').trim();
    const trackingToken = String(properties['_AI Tracking Token'] || '').trim();
    if (!taskId || !trackingToken) continue;
    // Match the old custom pixel's ID so overlapping installations are idempotent.
    tasks.set(taskId, {
      task_id: taskId,
      tracking_token: trackingToken,
      event_id: `${eventType}:${event.id}:${line.id || taskId}`,
      event_type: eventType,
      occurred_at: event.timestamp || '',
      source: 'shopify_app_pixel',
      shop_domain: shopDomain,
      // Shopify may redact email. Task/order association never depends on it.
      customer_email: checkout.email || '',
      visitor_id: event.clientId || '',
      variant_id: line.variant?.id || '',
      checkout_token: checkout.token || '',
    });
  }
  return [...tasks.values()];
}

export function createReporter({shopDomain, send = fetch, wait = ms => new Promise(resolve => setTimeout(resolve, ms))}) {
  const pending = new Map();
  const delivered = new Set();
  async function deliver(payload) {
    const key = `${payload.task_id}:${payload.event_id}`;
    if (delivered.has(key)) return;
    if (pending.has(key)) return pending.get(key);
    const request = (async () => {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const response = await send(ENDPOINT, {
            method: 'POST',
            headers: {'Content-Type': 'text/plain;charset=UTF-8'},
            credentials: 'omit',
            body: JSON.stringify(payload),
            keepalive: true,
          });
          if (response.ok) {
            delivered.add(key);
            if (delivered.size > 500) delivered.delete(delivered.values().next().value);
            return;
          }
          // Invalid or revoked task tokens are permanent failures, not retries.
          if (response.status !== 429 && response.status < 500) return;
        } catch (_) { /* A network failure can be retried with the same ID. */ }
        if (attempt < 2) await wait(500 * 2 ** attempt);
      }
    })();
    pending.set(key, request);
    try { await request; } finally { pending.delete(key); }
  }
  return async (eventType, event) => {
    if (!/^[a-z0-9][a-z0-9-]*\.myshopify\.com$/.test(shopDomain)) return;
    const payloads = checkoutPayloads(eventType, event, shopDomain);
    let next = 0;
    await Promise.all(Array.from({length: Math.min(3, payloads.length)}, async () => {
      while (next < payloads.length) await deliver(payloads[next++]);
    }));
  };
}
