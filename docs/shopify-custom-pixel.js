const API_BASE_URL = 'https://image-api.shopupup.com';

const propertiesToObject = (properties) => Object.fromEntries(
  (Array.isArray(properties) ? properties : [])
    .map((property) => [property.key, property.value])
);

const postEvent = async (payload) => {
  // text/plain keeps this a CORS "simple request" in Shopify's pixel sandbox.
  // The API still parses the JSON body with request.json().
  const response = await fetch(`${API_BASE_URL}/storefront-events`, {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain;charset=UTF-8' },
    body: JSON.stringify(payload),
    keepalive: true
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 300);
    throw new Error(`API returned ${response.status}: ${detail}`);
  }
};

const reportCheckoutEvent = async (eventType, event) => {
  const checkout = event.data?.checkout;
  if (!checkout) return;

  const trackedItems = new Map();
  for (const lineItem of (checkout.lineItems || [])) {
    const properties = propertiesToObject(lineItem.properties);
    const taskId = String(properties['_AI Task ID'] || '').trim();
    const trackingToken = String(properties['_AI Tracking Token'] || '').trim();
    if (!taskId || !trackingToken) continue;

    // A checkout can contain duplicate/restored lines for one generated image.
    // One idempotent event per task is enough and makes failures easy to inspect.
    trackedItems.set(taskId, { lineItem, taskId, trackingToken });
  }

  if (!trackedItems.size) {
    console.warn('[AI conversion tracking] no AI task properties found', {
      eventType,
      lineItemCount: (checkout.lineItems || []).length
    });
    return;
  }

  const requests = [...trackedItems.values()].map(async ({ lineItem, taskId, trackingToken }) => {
    try {
      await postEvent({
        task_id: taskId,
        tracking_token: trackingToken,
        event_id: `${eventType}:${event.id}:${lineItem.id || taskId}`,
        event_type: eventType,
        occurred_at: event.timestamp,
        source: 'shopify_web_pixel',
        customer_email: checkout.email || '',
        visitor_id: event.clientId || '',
        variant_id: lineItem.variant?.id || '',
        checkout_token: checkout.token || ''
      });
    } catch (error) {
      // Do not log the tracking token or customer data.
      console.error('[AI conversion tracking] checkout event failed', {
        eventType,
        taskId,
        message: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  });

  const results = await Promise.allSettled(requests);
  const failed = results.filter((result) => result.status === 'rejected').length;
  console.info('[AI conversion tracking] checkout event sent', {
    eventType,
    trackedTaskCount: trackedItems.size,
    failed
  });
};

analytics.subscribe('checkout_started', (event) => {
  reportCheckoutEvent('checkout_started', event);
});

analytics.subscribe('checkout_completed', (event) => {
  reportCheckoutEvent('checkout_completed', event);
});
