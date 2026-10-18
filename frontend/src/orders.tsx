import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { api, useData, useT, post, ErrorBox, Loading, Pick, safeUrl } from '@/lib/admin';

type Production = { id: number; status: string; costs?: {total?: string; currency?: string}; file_statuses?: {id: number; status: string}[]; files?: {line_id: string; file_url: string; dimensions?: number[]}[] };
type Summary = {id: string; name: string; createdAt: string; displayFinancialStatus: string; displayFulfillmentStatus: string};
type Item = {id: string; title: string; variantTitle: string; sku: string; unfulfilledQuantity: number; requiresShipping: boolean; images: {task_id: string; url: string}[]};
type Job = {state: string; detail?: string; production?: Production | null};
type Order = {id: string; name: string; cancelledAt: string | null; displayFinancialStatus: string; shippingAddress: Record<string, string> | null; items: Item[]; production: Production | null; job: Job};
type Choice = {line_id: string; task_id: string; variant_id: number; upscale: boolean; bleed: boolean};

export function PrintfulSettings() {
  const t = useT();
  const data = useData<{configured: boolean; store_id: string}>('/admin/api/production/settings');
  const [token, setToken] = useState('');
  const [store, setStore] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function save(test = false) {
    setBusy(true); setError('');
    try {
      if (test) {
        const result = await post<{stores: {id: number; name: string}[]}>('/admin/api/production/test', {});
        toast.success(t('连接成功：', 'Connected: ') + result.stores.map(s => `${s.name} (${s.id})`).join(', '));
      } else {
        await post('/admin/api/production/settings', {token, store_id: store ?? data.data?.store_id ?? ''});
        setToken(''); data.reload(); toast.success(t('已保存', 'Saved'));
      }
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  if (data.error) return <ErrorBox message={data.error} retry={data.reload}/>;
  if (!data.data) return <Loading/>;
  return <section className="max-w-2xl space-y-5 rounded-lg border p-6">
    <h2 className="text-lg font-semibold">Printful</h2>
    <p className="text-sm text-muted-foreground">{t('使用 Private Token 连接自己的店铺。Printful 也支持面向第三方用户的 OAuth 应用。', 'Use a Private Token for your own store. Printful also supports OAuth apps for third-party users.')} <a className="underline" href="https://developers.printful.com/tokens" target="_blank" rel="noreferrer">{t('获取 Token', 'Get a token')}</a></p>
    <p className="text-sm">{data.data.configured ? t('已配置密钥；留空保留原密钥。', 'Token configured; leave blank to keep it.') : t('尚未配置密钥。', 'No token configured.')}</p>
    <form className="space-y-5" onSubmit={e => {e.preventDefault(); void save();}}>
      <div className="space-y-2"><Label htmlFor="pf-token">Private Token</Label><Input id="pf-token" type="password" autoComplete="new-password" maxLength={4096} value={token} onChange={e => setToken(e.target.value)}/></div>
      <div className="space-y-2"><Label htmlFor="pf-store">Store ID</Label><Input id="pf-store" inputMode="numeric" pattern="[0-9]*" value={store ?? data.data.store_id} onChange={e => setStore(e.target.value)}/><p className="text-sm text-muted-foreground">{t('账户级 Token 必填；店铺级 Token 可留空。需允许读取店铺、商品和读写订单。', 'Required for account tokens; optional for store tokens. Allow store/product reads and order read/write access.')}</p></div>
      {error && <ErrorBox message={error}/>}
      <div className="flex gap-3"><Button disabled={busy}>{t('保存设置', 'Save settings')}</Button><Button type="button" variant="outline" disabled={busy || !data.data.configured} onClick={() => void save(true)}>{t('测试已保存的连接', 'Test saved connection')}</Button></div>
    </form>
  </section>;
}

export function Orders() {
  const t = useT();
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [cursors, setCursors] = useState<string[]>([]);
  const [selected, setSelected] = useState('');
  const data = useData<{nodes: Summary[]; pageInfo: {hasNextPage: boolean; endCursor: string}}>('/admin/api/orders?search=' + encodeURIComponent(query) + '&after=' + encodeURIComponent(cursors.at(-1) || ''));
  if (selected) return <OrderDetail key={selected} id={selected} back={() => setSelected('')}/>;
  return <div className="space-y-5">
    <div><h2 className="page-heading">{t('订单管理', 'Orders')}</h2><p className="page-description">{t('查看 Shopify 订单与原图，按商品推送到 Printful 草稿。', 'Review Shopify orders and original artwork, then send items to a Printful draft.')}</p></div>
    <form className="flex max-w-xl gap-2" onSubmit={e => {e.preventDefault(); setCursors([]); setQuery(search); data.reload();}}><Input aria-label={t('搜索订单', 'Search orders')} placeholder={t('订单号或 Shopify 搜索条件', 'Order number or Shopify search query')} value={search} onChange={e => setSearch(e.target.value)}/><Button>{t('搜索', 'Search')}</Button></form>
    {data.error ? <ErrorBox message={data.error} retry={data.reload}/> : data.loading ? <Loading/> : <>
      <div className="overflow-x-auto rounded-lg border"><table className="w-full text-left text-sm"><thead className="bg-muted"><tr>{[t('订单', 'Order'),t('时间', 'Created'),t('支付状态', 'Payment'),t('履约状态', 'Fulfillment'),''].map((v,i)=><th className="p-4" key={i}>{v}</th>)}</tr></thead><tbody>{data.data?.nodes.map(order=><tr key={order.id} className="border-t"><td className="p-4 font-medium">{order.name}</td><td className="p-4 whitespace-nowrap">{new Date(order.createdAt).toLocaleString()}</td><td className="p-4">{order.displayFinancialStatus}</td><td className="p-4">{order.displayFulfillmentStatus}</td><td className="p-4"><Button variant="outline" onClick={()=>setSelected(order.id.split('/').pop()!)}>{t('查看 / 推送', 'Review / send')}</Button></td></tr>)}</tbody></table>{!data.data?.nodes.length && <p className="p-8 text-center text-muted-foreground">{t('没有匹配的订单', 'No matching orders')}</p>}</div>
      <div className="flex gap-2"><Button variant="outline" disabled={!cursors.length} onClick={()=>setCursors(v=>v.slice(0,-1))}>{t('上一页', 'Previous')}</Button><Button variant="outline" disabled={!data.data?.pageInfo.hasNextPage} onClick={()=>setCursors(v=>[...v,data.data!.pageInfo.endCursor])}>{t('下一页', 'Next')}</Button></div>
    </>}
  </div>;
}

function OrderDetail({id, back}: {id: string; back: ()=>void}) {
  const t = useT();
  const data = useData<Order>('/admin/api/orders/' + id);
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [production, setProduction] = useState<Production | null>();
  const [running, setRunning] = useState(false);
  useEffect(() => {
    if (data.data?.job.state === 'running') setRunning(true);
    if (data.data?.job.state === 'failed') setError(data.data.job.detail || 'Processing failed');
  }, [data.data]);
  useEffect(() => {
    if (!running) return;
    let stopped = false;
    const timer = window.setInterval(() => {
      void api<Job>(`/admin/api/orders/${id}/production-job`).then(job => {
        if (stopped) return;
        if (job.state !== 'running') {
          setRunning(false); setProduction(job.production);
          if (job.state === 'failed') setError(job.detail || 'Processing failed');
          else toast.success(t('Printful 草稿已更新', 'Printful draft updated'));
        }
      }).catch(e => { if (!stopped) setError((e as Error).message); });
    }, 3000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [running, id]);
  async function send(refresh = false) {
    setBusy(true); setError('');
    try {
      if (refresh) {
        setProduction(await post<Production | null>(`/admin/api/orders/${id}/printful/status`, {}));
        toast.success(t('状态已刷新', 'Status refreshed'));
      } else {
        await post(`/admin/api/orders/${id}/printful`, {items:Object.values(choices)});
        setRunning(true);
      }
    } catch(e) {setError((e as Error).message);} finally {setBusy(false);}
  }
  const result = production === undefined ? data.data?.production : production;
  return <div className="space-y-5">
    <Button variant="outline" disabled={busy} onClick={back}>{t('返回订单', 'Back to orders')}</Button>
    {data.error ? <ErrorBox message={data.error} retry={data.reload}/> : !data.data ? <Loading/> : <>
      <div><h2 className="page-heading">{data.data.name}</h2><p className="page-description">{data.data.displayFinancialStatus} {data.data.cancelledAt && t(' · 已取消', ' · Cancelled')}</p></div>
      <section className="rounded-lg border p-5"><h3 className="mb-2 font-medium">{t('收货信息', 'Shipping address')}</h3><p className="text-sm">{data.data.shippingAddress ? Object.values(data.data.shippingAddress).filter(Boolean).join(' · ') : t('缺少收货地址', 'Shipping address missing')}</p></section>
      <p className="text-sm text-muted-foreground">{t('勾选需要推送的商品并核对 Printful Variant ID。更新草稿会以本次勾选商品替换草稿全部商品。放大会调用 FAL（可能收费）；出血为白色包边。', 'Select items and verify each Printful Variant ID. Updating a draft replaces all draft items with this selection. Upscaling uses FAL (charges may apply); bleed uses white wrap.')}</p>
      {data.data.items.map(item => {
        const choice = choices[item.id];
        const update = (value: Partial<Choice>) => setChoices(v=>({...v,[item.id]:{...v[item.id],...value}}));
        return <section key={item.id} className="rounded-lg border p-5 space-y-4">
          <div className="flex items-start justify-between gap-4"><div><h3 className="font-medium">{item.title}</h3><p className="text-sm text-muted-foreground">{item.variantTitle} · SKU {item.sku || '—'} · {t('未履约数量', 'Unfulfilled quantity')} {item.unfulfilledQuantity}</p></div><Label className="flex shrink-0 gap-2 whitespace-nowrap"><Switch disabled={busy || running || !item.images.length || !item.requiresShipping || !item.unfulfilledQuantity || !!data.data?.cancelledAt} checked={!!choice} onCheckedChange={on=>setChoices(v=>{const next={...v}; if(on) next[item.id]={line_id:item.id,task_id:item.images[0].task_id,variant_id:0,upscale:true,bleed:true}; else delete next[item.id];return next;})}/>{t('推送', 'Send')}</Label></div>
          {!item.images.length && <p className="text-sm text-muted-foreground">{t('未关联可用的无水印原图', 'No original artwork available')}</p>}
          <div className="flex flex-wrap gap-3">{item.images.map(image=><a href={safeUrl(image.url)} key={image.task_id} target="_blank" rel="noreferrer"><img className="h-32 w-32 rounded border object-contain" src={safeUrl(image.url)} alt={item.title}/></a>)}</div>
          {choice && <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-2"><Label>{t('原图', 'Artwork')}</Label><Pick disabled={busy} label={t('原图', 'Artwork')} value={choice.task_id} options={item.images.map(i=>[i.task_id,i.task_id])} onChange={task_id=>update({task_id})}/></div>
            <div className="space-y-2"><Label htmlFor={'variant-'+item.id}>Printful Variant ID</Label><Input id={'variant-'+item.id} disabled={busy} type="number" min="1" value={choice.variant_id || ''} onChange={e=>update({variant_id:Number(e.target.value)})}/><p className="text-xs text-muted-foreground">{t('出血支持画布 ID：5、6、7、19316。请按实际商品核对。', 'Canvas bleed supported for IDs 5, 6, 7, 19316. Verify the actual product.')}</p></div>
            <Label className="flex items-center gap-2"><Switch disabled={busy} checked={choice.upscale} onCheckedChange={upscale=>update({upscale})}/>{t('AuraSR 放大 4×', 'AuraSR upscale 4×')}</Label>
            <Label className="flex items-center gap-2"><Switch disabled={busy} checked={choice.bleed} onCheckedChange={bleed=>update({bleed})}/>{t('添加白色出血', 'Add white bleed')}</Label>
          </div>}
        </section>;
      })}
      {error && <ErrorBox message={error}/>}
      <div className="flex flex-wrap gap-3"><Button disabled={busy || running || !Object.keys(choices).length || Object.values(choices).some(c=>!c.variant_id) || !!data.data.cancelledAt || !data.data.shippingAddress} onClick={()=>void send()}>{busy || running ? t('处理中，请稍候…', 'Processing, please wait…') : t('推送到 Printful 草稿', 'Send to Printful draft')}</Button><Button variant="outline" disabled={busy} onClick={()=>void send(true)}>{t('查询 Printful 状态', 'Check Printful status')}</Button></div>
      <p className="text-sm text-muted-foreground">{t('草稿不会自动扣款生产。请在 Printful 核对印刷文件、运费和总价后确认。', 'Drafts do not start production. Review print files, shipping and total in Printful before confirming.')}</p>
      {result && <section className="space-y-2 rounded-lg border p-5"><h3 className="font-medium">Printful #{result.id} · {result.status}</h3><p>{result.costs?.total} {result.costs?.currency}</p>{result.file_statuses?.map((f,i)=><p key={i} className="text-sm">{t('文件', 'File')} #{f.id}: {f.status}</p>)}{result.files?.map((f,i)=><p key={i} className="text-sm"><a className="underline" href={safeUrl(f.file_url)} target="_blank" rel="noreferrer">{t('查看印刷文件', 'View print file')} {i+1}</a> {f.dimensions?.join(' × ')}</p>)}</section>}
    </>}
  </div>;
}
