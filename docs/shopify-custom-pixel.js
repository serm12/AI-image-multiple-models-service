const API_BASE_URL = 'https://image-api.shopupup.com';

const propertiesToObject = (properties = []) => Object.fromEntries(
  properties.map((property) => [property.key, property.value])
);

const reportCheckoutEvent = async (eventType, event) => {
  const checkout = event.data?.checkout;
  if (!checkout) return;

  const requests = (checkout.lineItems || []).flatMap((lineItem) => {
    const properties = propertiesToObject(lineItem.properties);
    const taskId = String(properties['_AI Task ID'] || '').trim();
    const trackingToken = String(properties['_AI Tracking Token'] || '').trim();
    if (!taskId || !trackingToken) return [];

    return [fetch(`${API_BASE_URL}/storefront-events`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_id: taskId,
        tracking_token: trackingToken,
        event_id: `${eventType}:${event.id}:${lineItem.id || taskId}`,
        event_type: eventType,
        occurred_at: event.timestamp,
        source: 'shopify_web_pixel',
        customer_email: checkout.email || '',
        visitor_id: event.clientId || '',
        variant_id: lineItem.merchandise?.id || '',
        checkout_token: checkout.token || ''
      }),
      keepalive: true
    })];
  });

  await Promise.allSettled(requests);
};

analytics.subscribe('checkout_started', (event) => {
  reportCheckoutEvent('checkout_started', event);
});

analytics.subscribe('checkout_completed', (event) => {
  reportCheckoutEvent('checkout_completed', event);
});
