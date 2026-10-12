import '@shopify/ui-extensions/preact';
import {render} from 'preact';
import {useEffect, useState} from 'preact/hooks';

type Item = {line_item_id: string; task_id: string; title: string;
  image: {url: string; download_url: string} | null};
type Result = {order_name: string; items: Item[]};
export default () => render(<Originals />, document.body);

function Originals() {
  const orderId = shopify.order.value?.id?.split('/').pop() || '';
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [login, setLogin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setResult(null); setError(''); setLoading(true); setLogin(false);
    (async () => {
      try {
        if (!/^\d+$/.test(orderId)) throw new Error('Your order is being prepared. Please refresh shortly.');
        const token = await shopify.sessionToken.get();
        const response = await fetch(`https://image-api.shopupup.com/shopify/customer/orders/${orderId}/images`, {
          headers: {Authorization: `Bearer ${token}`}, signal: controller.signal,
        });
        if (response.status === 401) {
          if (active) setLogin(true);
          throw new Error('Please sign in to access your original artwork.');
        }
        if (!response.ok) throw new Error(response.status === 404
          ? 'Your artwork is not available yet. Please refresh shortly or contact us for help.'
          : 'We could not load your artwork. Please try again.');
        const data: Result = await response.json();
        if (active) setResult(data);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : 'Please try again.');
      } finally {if (active) setLoading(false);}
    })();
    return () => {active = false; controller.abort();};
  }, [orderId, revision]);

  return <s-section heading="Your original artwork">
    <s-stack gap="base">
      <s-text>Download the watermark-free images from your Paintingify order.</s-text>
      {loading && <s-spinner accessibilityLabel="Loading your artwork" />}
      {error && <s-banner tone="info">{error}</s-banner>}
      {login && <s-button onClick={async () => {
        try {await shopify.requireLogin(); setRevision(value => value + 1);}
        catch {setError('Please sign in and refresh this page.');}
      }}>Sign in to download</s-button>}
      {result?.items.map(item => <s-box key={`${item.line_item_id}-${item.task_id}`} padding="base" border="base" borderRadius="base">
        <s-stack gap="base">
          <s-heading>{item.title}</s-heading>
          {item.image ? <>
            <s-box maxInlineSize="320px"><s-image src={item.image.url} alt={`${item.title} — original artwork`} objectFit="contain" /></s-box>
            <s-button href={item.image.download_url} target="_blank" variant="primary">Download original image</s-button>
            <s-link href={item.image.url} target="_blank">View full-size image</s-link>
          </> : <s-text>Your original image is being prepared. Please check back soon.</s-text>}
        </s-stack>
      </s-box>)}
      {result?.items.length === 0 && <s-text>No downloadable artwork is linked to this order.</s-text>}
      <s-button disabled={loading} onClick={() => setRevision(value => value + 1)}>Refresh images</s-button>
    </s-stack>
  </s-section>;
}
