import '@shopify/ui-extensions/preact';
import {render} from 'preact';
import {useEffect, useState} from 'preact/hooks';

type Item = {line_item_id: string; title: string; task_id: string;
  image: {url: string; filename: string} | null};
type OrderImages = {order_name: string; items: Item[]};

export default () => render(<OrderOriginals />, document.body);

function OrderOriginals() {
  const selected = shopify.data.selected?.[0]?.id || '';
  const [result, setResult] = useState<OrderImages | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil((result?.items.length || 0) / 2));
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError('');
    setResult(null);
    setPage(0);
    (async () => {
      try {
        const orderId = selected.split('/').pop();
        if (!orderId || !/^\d+$/.test(orderId)) throw new Error('未获取到当前订单，请重新打开订单页。');
        const token = await shopify.auth.idToken();
        if (!token) throw new Error('请重新打开应用以验证登录。');
        const response = await fetch(`https://image-api.shopupup.com/shopify/orders/${orderId}/images`, {
          headers: {Authorization: `Bearer ${token}`}, signal: controller.signal,
        });
        if (!response.ok) throw new Error(response.status === 404
          ? '订单不存在或当前账户没有访问权限。' : '暂时无法读取原图，请点击刷新重试。');
        const data: OrderImages = await response.json();
        if (active) setResult(data);
      } catch (e) {
        if (active) setError(e instanceof Error ? e.message : '读取失败，请重试。');
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {active = false; controller.abort();};
  }, [selected, revision]);

  return <s-admin-block heading="Paintingify · 无水印原图">
    <s-stack direction="block" gap="base">
      <s-button onClick={() => setRevision(value => value + 1)} disabled={loading}>刷新原图</s-button>
      {loading && <s-text>正在读取订单对应的原图…</s-text>}
      {error && <s-banner tone="critical">{error}</s-banner>}
      {!loading && !error && result?.items.length === 0 && <s-text>此订单尚未关联 AI 生成图片。</s-text>}
      {result && <s-text>订单 {result.order_name} · 共 {result.items.length} 张原图</s-text>}
      {pageCount > 1 && <s-stack direction="inline" gap="base">
        <s-button disabled={page === 0} onClick={() => setPage(value => value - 1)}>上一页</s-button>
        <s-text>{page + 1} / {pageCount}</s-text>
        <s-button disabled={page + 1 >= pageCount} onClick={() => setPage(value => value + 1)}>下一页</s-button>
      </s-stack>}
      {result?.items.slice(page * 2, page * 2 + 2).map(item => <s-box key={`${item.line_item_id}-${item.task_id}`} padding="small" borderWidth="base" borderRadius="base">
        <s-grid gridTemplateColumns="80px minmax(0, 1fr)" gap="base" alignItems="center">
          <s-box>{item.image && <s-image src={item.image.url} alt={`${item.title} 无水印原图`} inlineSize="fill" aspectRatio="1/1" objectFit="contain" />}</s-box>
        <s-stack direction="block" gap="small">
          <s-text type="strong">{item.title}</s-text>
          {item.image ? <>
            <s-stack direction="inline" gap="base">
              <s-link href={item.image.url} target="_blank">打开无水印原图</s-link>
              <s-link href={item.image.url} download={item.image.filename} target="_blank">下载原图</s-link>
            </s-stack>
          </> : <s-text>此任务的无水印原图暂不可用。</s-text>}
        </s-stack>
        </s-grid>
      </s-box>)}
    </s-stack>
  </s-admin-block>;
}
