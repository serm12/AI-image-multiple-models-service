import { usePageBreadcrumb } from './workspace-header';
import { createPortal } from 'react-dom';
import { ArrowLeft, Search, RefreshCw, ChevronLeft, ChevronRight, ArrowUpRight, ImageIcon } from 'lucide-react';
import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { api, useData, useT, post, ErrorBox, Loading, Pick, safeUrl } from '@/lib/admin';

type Production = { id: number; status: string; costs?: {total?: string; currency?: string}; file_statuses?: {id: number; status: string}[]; files?: {line_id: string; file_url: string; dimensions?: number[]}[] };
type LinkedOrder = {id: number; external_id?: string; status: string; dashboard_url?: string; shipments?: {tracking_number?: string; tracking_url?: string; carrier?: string}[]};
type PrintfulLinks = {printful_orders?: LinkedOrder[]; printful_sync?: {state: string}};
type StatusTone = 'success' | 'neutral' | 'attention' | 'warning' | 'critical' | 'info';
function StatusLabel({label, tone = 'neutral', complete = false}: {label: string; tone?: StatusTone; complete?: boolean}) {
  const colors = {success: 'bg-[#dcfce7] text-[#166534]', neutral: 'bg-[#f1f1f1] text-[#616161]', attention: 'bg-[#ffeb78] text-[#4f4700]', warning: 'bg-[#ffebd5] text-[#5e4200]', critical: 'bg-[#fee9e8] text-[#8e1f0b]', info: 'bg-[#eaf4ff] text-[#00527c]'};
  return <span className={'inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-medium leading-4 ' + colors[tone]}><span aria-hidden="true" className={'size-2 shrink-0 rounded-[2px] ' + (complete ? 'bg-current' : 'border border-current')}/>{label}</span>;
}
function ShopifyStatus({status, kind}: {status?: string | null; kind: 'payment' | 'fulfillment'}) {
  const t = useT();
  const payment: Record<string, [string, string, StatusTone, boolean?]> = {
    PAID: ['已付款', 'Paid', 'neutral', true], AUTHORIZED: ['已授权', 'Authorized', 'neutral'], PENDING: ['待付款', 'Payment pending', 'attention'], PARTIALLY_PAID: ['部分付款', 'Partially paid', 'attention'], PARTIALLY_REFUNDED: ['部分退款', 'Partially refunded', 'neutral'], REFUNDED: ['已退款', 'Refunded', 'neutral', true], VOIDED: ['已作废', 'Voided', 'neutral'], EXPIRED: ['已过期', 'Expired', 'critical'],
  };
  const fulfillment: Record<string, [string, string, StatusTone, boolean?]> = {
    FULFILLED: ['已发货', 'Fulfilled', 'success', true], UNFULFILLED: ['未发货', 'Unfulfilled', 'attention'], PARTIALLY_FULFILLED: ['部分发货', 'Partially fulfilled', 'attention'], IN_PROGRESS: ['处理中', 'In progress', 'info'], ON_HOLD: ['暂停发货', 'On hold', 'warning'], SCHEDULED: ['已安排发货', 'Scheduled', 'info'], FULFILLMENT_NOT_REQUIRED: ['无需发货', 'Not required', 'neutral'], REQUEST_DECLINED: ['发货请求被拒绝', 'Request declined', 'critical'], OPEN: ['未发货', 'Unfulfilled', 'attention'], PENDING_FULFILLMENT: ['待处理发货', 'Pending fulfillment', 'info'], RESTOCKED: ['已重新入库', 'Restocked', 'neutral'],
  };
  const value = (kind === 'payment' ? payment : fulfillment)[status || ''];
  return <StatusLabel label={value ? t(value[0], value[1]) : t('待核对', 'Needs review')} tone={value?.[2]} complete={value?.[3]}/>;
}
function PrintfulStatus({status}: {status: string}) {
  const t = useT();
  const labels: Record<string, string> = {fulfilled: t('已发货', 'Fulfilled'), partial: t('部分发货', 'Partially fulfilled'), draft: t('草稿', 'Draft'), pending: t('待处理', 'Pending'), inprocess: t('生产中', 'In production'), onhold: t('暂停', 'On hold'), failed: t('失败', 'Failed'), canceled: t('已取消', 'Canceled'), archived: t('已归档', 'Archived'), inreview: t('审核中', 'In review')};
  const tone = status === 'fulfilled' ? 'success' : status === 'failed' || status === 'canceled' ? 'critical' : status === 'onhold' ? 'warning' : status === 'partial' || status === 'pending' ? 'attention' : status === 'inprocess' || status === 'inreview' ? 'info' : 'neutral';
  return <StatusLabel label={labels[status] || t('待核对', 'Needs review')} tone={tone} complete={status === 'fulfilled'}/>;
}

type Contact = {email?: string | null; customer?: {firstName?: string | null; lastName?: string | null; defaultEmailAddress?: {emailAddress?: string | null} | null} | null; shippingAddress?: {name?: string | null} | null};
type OrderType = "physical" | "digital" | "mixed" | "unknown";
type Summary = Contact & PrintfulLinks & {product_summary?: {title: string; count: number; quantity: number; image_url?: string | null};order_type: OrderType;id: string; name: string; createdAt: string; displayFinancialStatus: string; displayFulfillmentStatus: string};
type Item = {item_type: OrderType;id: string; title: string; variantTitle: string; sku: string; currentQuantity: number; requiresShipping: boolean; images: {task_id: string; url: string}[]};
type Job = {state: string; detail?: string; production?: Production | null};
type Order = Contact & PrintfulLinks & {order_type: OrderType;id: string; name: string; cancelledAt: string | null; displayFinancialStatus: string; displayFulfillmentStatus: string; shippingAddress: Record<string, string> | null; items: Item[]; production: Production | null; job: Job};
type Choice = {line_id: string; task_id: string; variant_id: number; upscale: boolean; bleed: boolean};

function TypeBadge({kind}: {kind: OrderType}) {
  const t = useT();
  const labels = {physical: t('实物', 'Physical'), digital: t('虚拟', 'Digital'), mixed: t('混合订单', 'Mixed'), unknown: t('待核对', 'Needs review')};
  const colors = {physical: 'bg-blue-50 text-blue-700', digital: 'bg-violet-50 text-violet-700', mixed: 'bg-amber-50 text-amber-800', unknown: 'bg-slate-100 text-slate-600'};
  return <span className={'inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium ' + (colors[kind] || colors.unknown)}>{labels[kind] || labels.unknown}</span>;
}

function CustomerContact({order}: {order: Contact}) {
  const t = useT();
  const name = [order.customer?.firstName, order.customer?.lastName].filter(Boolean).join(' ').trim() || order.shippingAddress?.name;
  const email = order.email || order.customer?.defaultEmailAddress?.emailAddress;
  return <div className="min-w-40 space-y-1"><p className="font-medium">{name || t('未提供姓名', 'Name not provided')}</p><p className="break-all text-sm text-muted-foreground">{email || t('未提供邮箱', 'Email not provided')}</p></div>;
}

export function PrintfulSettings() {
  const t = useT();
  const data = useData<{configured: boolean; store_id: string; connection_status: string}>('/admin/api/production/settings');
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
    } catch (e) { setError((e as Error).message); } finally { data.reload(); setBusy(false); }
  }
  if (data.error) return <ErrorBox message={data.error} retry={data.reload}/>;
  if (!data.data) return <Loading/>;
  const changed = !!token.trim() || (store !== undefined && store !== data.data.store_id);
  const status = changed ? 'unverified' : data.data.connection_status;
  const connected = data.data.configured && status === 'connected';
  return <section className="max-w-2xl space-y-5 rounded-xl border bg-white p-6 shadow-sm">
    <h2 className="text-lg font-semibold">{t('Printful Token 配置', 'Printful token settings')}</h2>
    <p className="text-sm text-muted-foreground">{t('使用 Private Token 连接自己的店铺。Printful 也支持面向第三方用户的 OAuth 应用。', 'Use a Private Token for your own store. Printful also supports OAuth apps for third-party users.')} <a className="underline" href="https://developers.printful.com/tokens" target="_blank" rel="noreferrer">{t('获取 Token', 'Get a token')}</a></p>
    <div role="status" className={'inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ' + (connected ? 'bg-emerald-50 text-emerald-700' : status === 'failed' ? 'bg-red-50 text-red-700' : 'bg-slate-100 text-slate-600')}><span aria-hidden="true" className={'size-2 rounded-full ' + (connected ? 'bg-emerald-500' : status === 'failed' ? 'bg-red-500' : 'bg-slate-400')} />{connected ? t('连接成功', 'Connected') : status === 'failed' ? t('连接失败，请重新测试', 'Connection failed; test again') : data.data.configured ? t('已配置 · 待验证', 'Configured · Not verified') : t('未配置', 'Not configured')}</div>
    <p className="text-sm">{data.data.configured ? t('已配置密钥；留空保留原密钥。', 'Token configured; leave blank to keep it.') : t('尚未配置密钥。', 'No token configured.')}</p>
    <form className="space-y-5" onSubmit={e => {e.preventDefault(); void save();}}>
      <div className="space-y-2"><Label htmlFor="pf-token">Printful API Token</Label><Input id="pf-token" placeholder={t('粘贴你的 Printful Private Token', 'Paste your Printful Private Token')} type="password" autoComplete="new-password" maxLength={4096} value={token} onChange={e => setToken(e.target.value)}/></div>
      <div className="space-y-2"><Label htmlFor="pf-store">Store ID</Label><Input id="pf-store" inputMode="numeric" pattern="[0-9]*" value={store ?? data.data.store_id} onChange={e => setStore(e.target.value)}/><p className="text-sm text-muted-foreground">{t('账户级 Token 必填；店铺级 Token 可留空。需允许读取店铺、商品和读写订单。', 'Required for account tokens; optional for store tokens. Allow store/product reads and order read/write access.')}</p></div>
      {error && <ErrorBox message={error}/>}
      <div className="flex gap-3"><Button disabled={busy}>{t('保存设置', 'Save settings')}</Button><Button type="button" variant="outline" disabled={busy || !data.data.configured || changed} onClick={() => void save(true)}>{t('测试已保存的连接', 'Test saved connection')}</Button></div>
    </form>
  </section>;
}

export function Orders() {
  const t = useT();
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [cursors, setCursors] = useState<string[]>([]);
  const [selected, setSelected] = useState(()=>{const id=new URLSearchParams(location.search).get('order')||'';return /^\d+$/.test(id)?id:'';});
  useEffect(()=>{const back=()=>{setSelected('');history.replaceState(null,'',location.pathname);};window.addEventListener('admin:orders-back',back);return ()=>window.removeEventListener('admin:orders-back',back);},[]);
  const filters = [
    {id: 'all', label: t('全部', 'All'), query: ''},
    {id: 'shipped', label: t('已发货', 'Fulfilled'), query: 'fulfillment_status:shipped AND -status:cancelled'},
    {id: 'pending', label: t('待发货', 'Unfulfilled'), query: 'fulfillment_status:unshipped AND -status:cancelled'},
    {id: 'cancelled', label: t('已取消', 'Cancelled'), query: 'status:cancelled'},
  ];
  const filterQuery = filters.find(f=>f.id===filter)!.query;
  const combined = [query.trim() ? '(' + query.trim() + ')' : '', filterQuery].filter(Boolean).join(' AND ');
  const data = useData<{nodes: Summary[]; pageInfo: {hasNextPage: boolean; endCursor: string}}>('/admin/api/orders?search=' + encodeURIComponent(combined) + '&after=' + encodeURIComponent(cursors.at(-1) || ''));
  function openOrder(order: Summary) { setSelected(order.id.split('/').pop()!); }
  if (selected) return <OrderDetail key={selected} id={selected} back={() => {setSelected('');history.replaceState(null,'',location.pathname);}}/>;
  return <section className="overflow-hidden rounded-3xl border border-neutral-200/80 bg-white">
    <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-5 sm:px-6">
      <div role="group" aria-label={t('订单状态筛选', 'Filter order status')} className="inline-flex max-w-full flex-wrap gap-1 rounded-full bg-neutral-100 p-1">{filters.map(f=><button type="button" key={f.id} aria-pressed={filter===f.id} onClick={()=>{setFilter(f.id);setCursors([]);}} className={'rounded-full px-4 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 ' + (filter===f.id ? 'bg-white font-medium text-neutral-900 shadow-sm' : 'text-neutral-500 hover:text-neutral-900')}>{f.label}</button>)}</div>
      <div className="flex w-full items-center gap-2 sm:w-auto"><form className="flex min-w-0 flex-1 items-center rounded-full bg-neutral-100 pl-3 sm:w-64" onSubmit={e=>{e.preventDefault();setCursors([]);setQuery(search);if(search===query)data.reload();}}><Search aria-hidden="true" className="size-4 shrink-0 text-neutral-500"/><Input className="h-9 min-w-0 border-0 bg-transparent! shadow-none focus-visible:ring-0" aria-label={t('搜索订单', 'Search orders')} placeholder={t('搜索订单…', 'Search orders…')} value={search} onChange={e=>setSearch(e.target.value)}/><button type="submit" className="mr-1 shrink-0 whitespace-nowrap rounded-full px-3 py-1 text-xs text-neutral-600 hover:bg-white focus-visible:outline-2">{t('搜索', 'Search')}</button></form><Button type="button" variant="outline" size="icon" className="rounded-full border-neutral-200" disabled={data.loading} aria-label={t('刷新订单', 'Refresh orders')} onClick={data.reload}><RefreshCw className={'size-4 ' + (data.loading ? 'animate-spin motion-reduce:animate-none' : '')}/></Button></div>
    </div>
    {data.error ? <div className="p-6"><ErrorBox message={data.error} retry={data.reload}/></div> : data.loading ? <div className="p-8"><Loading/></div> : <>
      <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="border-y border-neutral-200/80 bg-neutral-100/90 text-xs font-normal text-neutral-500"><tr>{[t('订单', 'Order'),t('商品', 'Product'),t('客户', 'Customer'),t('日期', 'Date'),t('支付状态', 'Payment'),t('发货状态', 'Fulfillment'),'Printful',''].map((label,i)=><th key={i} className="whitespace-nowrap px-5 py-3 font-normal first:pl-6 last:pr-6">{label}</th>)}</tr></thead><tbody className="divide-y divide-neutral-200/80">{data.data?.nodes.map(order=>{
        const product = order.product_summary;
        return <tr key={order.id} className="transition-colors hover:bg-neutral-50/80">
          <td className="whitespace-nowrap py-4 pl-6 pr-5"><button className="font-medium text-neutral-900 hover:underline focus-visible:outline-2" onClick={()=>openOrder(order)}>{order.name}</button></td>
          <td className="px-5 py-3"><button className="flex w-64 items-center gap-3 text-left focus-visible:outline-2" onClick={()=>openOrder(order)}>{product?.image_url ? <img src={safeUrl(product.image_url)} alt="" loading="lazy" className="size-11 shrink-0 rounded-xl bg-neutral-100 object-cover"/> : <span className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-neutral-100 text-neutral-400"><ImageIcon className="size-5"/></span>}<span className="min-w-0"><span className="block truncate font-medium text-neutral-800" title={product?.title}>{product?.title || t('查看订单商品', 'View order items')}</span><span className="mt-1 flex items-center gap-2"><TypeBadge kind={order.order_type}/>{product && <span className="text-xs text-neutral-500">{product.quantity} {t('件商品', 'items')}</span>}</span></span></button></td>
          <td className="px-5 py-3"><div className="max-w-52 text-[13px]"><CustomerContact order={order}/></div></td>
          <td className="whitespace-nowrap px-5 py-3 text-xs text-neutral-500"><time dateTime={order.createdAt} title={new Date(order.createdAt).toLocaleString()}>{new Date(order.createdAt).toLocaleDateString(undefined,{year:'numeric',month:'2-digit',day:'2-digit'})}</time></td>
          <td className="px-5 py-3"><ShopifyStatus kind="payment" status={order.displayFinancialStatus}/></td>
          <td className="px-5 py-3"><ShopifyStatus kind="fulfillment" status={order.displayFulfillmentStatus}/></td>
          <td className="px-5 py-3"><div className="flex min-w-20 flex-wrap gap-1">{order.printful_orders?.map(p=><span key={p.id} title={p.external_id}><PrintfulStatus status={p.status}/></span>)}{!order.printful_orders?.length && <span className="whitespace-nowrap text-xs text-neutral-400">{order.printful_sync?.state === 'failed' ? t('同步失败', 'Sync failed') : order.printful_sync?.state === 'unconfigured' ? t('未配置', 'Not configured') : t('未关联', 'Not linked')}</span>}</div></td>
          <td className="py-3 pl-3 pr-6"><Button variant="ghost" size="icon" className="rounded-full" aria-label={t('查看订单 ', 'View order ') + order.name} title={t('查看详情', 'View details')} onClick={()=>openOrder(order)}><ArrowUpRight className="size-4"/></Button></td>
        </tr>;
      })}</tbody></table>{!data.data?.nodes.length && <div className="px-6 py-16 text-center"><p className="text-sm text-neutral-600">{t('没有匹配的订单', 'No matching orders')}</p><Button variant="ghost" className="mt-3" onClick={()=>{setSearch('');setQuery('');setFilter('all');setCursors([]);}}>{t('清除筛选', 'Clear filters')}</Button></div>}</div>
      <footer className="flex items-center justify-between gap-4 border-t border-neutral-200/80 px-6 py-4"><span className="text-xs text-neutral-500">{t('本页', 'Showing')} {data.data?.nodes.length || 0} {t('笔订单', 'orders')}</span><div className="flex items-center gap-2"><Button size="icon" variant="outline" className="size-8 rounded-full" aria-label={t('上一页', 'Previous page')} disabled={!cursors.length} onClick={()=>setCursors(v=>v.slice(0,-1))}><ChevronLeft className="size-4"/></Button><span className="min-w-6 text-center text-xs tabular-nums text-neutral-600">{cursors.length+1}</span><Button size="icon" variant="outline" className="size-8 rounded-full" aria-label={t('下一页', 'Next page')} disabled={!data.data?.pageInfo.hasNextPage} onClick={()=>setCursors(v=>[...v,data.data!.pageInfo.endCursor])}><ChevronRight className="size-4"/></Button></div></footer>
    </>}
  </section>;
}

function OrderDetail({id, back}: {id: string; back: ()=>void}) {
  const t = useT();
  const data = useData<Order>('/admin/api/orders/' + id);
  usePageBreadcrumb(data.data?.name || t('订单详情','Order details'));
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
        await post(`/admin/api/orders/${id}/printful/status`, {});
        data.reload();
        toast.success(t('状态已刷新', 'Status refreshed'));
      } else {
        await post(`/admin/api/orders/${id}/printful`, {items:Object.values(choices)});
        setRunning(true);
      }
    } catch(e) {setError((e as Error).message);} finally {setBusy(false);}
  }
  const result = production === undefined ? data.data?.production : production;
  const linked = data.data?.printful_orders || [];
  const duplicate = linked.length > 1 || linked.some(p => p.status !== 'draft');
  const backSlot = document.getElementById('order-detail-back');
  const backButton = <Button type="button" variant="outline" size="sm" disabled={busy} onClick={back}><ArrowLeft className="size-4" />{t('返回订单', 'Back to orders')}</Button>;
  return <div className="space-y-5">
    {backSlot ? createPortal(backButton, backSlot) : backButton}
    {data.error ? <ErrorBox message={data.error} retry={data.reload}/> : !data.data ? <Loading/> : <>
      <div><div className="flex items-center gap-3"><h2 className="page-heading">{data.data.name}</h2><TypeBadge kind={data.data.order_type}/></div><div className="mt-2 flex flex-wrap gap-2"><ShopifyStatus kind="payment" status={data.data.displayFinancialStatus}/><ShopifyStatus kind="fulfillment" status={data.data.displayFulfillmentStatus}/>{data.data.cancelledAt && <StatusLabel label={t('已取消', 'Canceled')} tone="critical"/>}</div></div>
      <section className="rounded-xl border bg-white p-5 shadow-sm space-y-3"><div className="flex items-center justify-between gap-3"><h3 className="font-medium">{t('Printful 关联订单', 'Linked Printful orders')}</h3><Button variant="outline" disabled={busy} onClick={()=>void send(true)}>{t('同步状态', 'Sync status')}</Button></div>
        <p className="text-xs text-muted-foreground">{t('打开页面自动读取 Printful 状态（缓存最多 60 秒）；同步状态可立即刷新。', 'Status loads on opening (cached up to 60 seconds); sync to refresh immediately.')}</p>
        {data.data.printful_sync?.state === 'failed' && <p className="text-sm text-red-700">{t('Printful 同步失败，请重试。推送前会再次检查已有订单。', 'Printful sync failed. Retry; existing orders are checked again before sending.')}</p>}
        {!linked.length && <p className="text-sm text-muted-foreground">{data.data.printful_sync?.state === 'synced' ? t('未找到关联的 Printful 订单。', 'No linked Printful orders found.') : t('尚未同步到关联订单。', 'Linked orders have not been synced.')}</p>}
        {linked.map(p=><div key={p.id} className="space-y-2 border-t pt-3"><div className="flex flex-wrap items-center gap-3"><a className="underline" href={safeUrl(p.dashboard_url || 'https://www.printful.com/dashboard?order_id='+p.id)} target="_blank" rel="noreferrer">Printful #{p.id}</a><PrintfulStatus status={p.status}/></div><p className="break-all text-sm text-muted-foreground">{p.external_id}</p>{p.shipments?.map((shipment,i)=><p key={i} className="text-sm">{shipment.carrier} {shipment.tracking_url ? <a className="underline" href={safeUrl(shipment.tracking_url)} target="_blank" rel="noreferrer">{shipment.tracking_number || t('物流追踪', 'Track shipment')}</a> : shipment.tracking_number}</p>)}</div>)}
        {duplicate && <p className="text-sm text-amber-700">{t('已有生产、历史或多笔关联订单，已阻止重复推送。请先在 Printful 核对，补印须单独处理。', 'Existing production, historical or multiple orders found. Duplicate submission is blocked; review in Printful. Reprints need separate handling.')}</p>}
      </section>
      <section className="rounded-xl border bg-white p-5 shadow-sm"><h3 className="mb-2 font-medium">{t('客户信息', 'Customer information')}</h3><CustomerContact order={data.data}/></section>
      <section className="rounded-xl border bg-white p-5 shadow-sm"><h3 className="mb-2 font-medium">{t('收货信息', 'Shipping address')}</h3><p className="text-sm">{data.data.shippingAddress ? Object.values(data.data.shippingAddress).filter(Boolean).join(' · ') : t('缺少收货地址', 'Shipping address missing')}</p></section>
      {data.data.order_type === 'digital' && <p className="rounded-xl border bg-white p-5 text-sm">{t('虚拟订单，仅交付电子图片，无需推送 Printful 生产配送。', 'Digital order: deliver image files only. No Printful production or shipping is needed.')}</p>}
      {data.data.order_type === 'mixed' && <p className="rounded-xl border bg-white p-5 text-sm">{t('混合订单：仅实物商品可推送 Printful，虚拟商品保留电子交付。', 'Mixed order: only physical items can be sent to Printful; digital items are delivered electronically.')}</p>}
      {data.data.order_type !== 'digital' && <p className="text-sm text-muted-foreground">{t('勾选需要推送的商品并核对 Printful Variant ID。更新草稿会以本次勾选商品替换草稿全部商品。放大会调用 FAL（可能收费）；出血为白色包边。', 'Select items and verify each Printful Variant ID. Updating a draft replaces all draft items with this selection. Upscaling uses FAL (charges may apply); bleed uses white wrap.')}</p>}
      {data.data.items.map(item => {
        const choice = choices[item.id];
        const update = (value: Partial<Choice>) => setChoices(v=>({...v,[item.id]:{...v[item.id],...value}}));
        return <section key={item.id} className="rounded-xl border bg-white p-5 shadow-sm space-y-4">
          <div className="flex items-start justify-between gap-4"><div><div className="flex flex-wrap items-center gap-2"><h3 className="font-medium">{item.title}</h3><TypeBadge kind={item.item_type}/></div><p className="text-sm text-muted-foreground">{item.variantTitle} · SKU {item.sku || '—'}</p></div>{item.item_type === 'physical' && <Label className="flex shrink-0 gap-2 whitespace-nowrap"><Switch disabled={busy || running || duplicate || !item.images.length || item.item_type !== 'physical' || !item.requiresShipping || item.currentQuantity < 1 || !!data.data?.cancelledAt} checked={!!choice} onCheckedChange={on=>setChoices(v=>{const next={...v}; if(on) next[item.id]={line_id:item.id,task_id:item.images[0].task_id,variant_id:0,upscale:true,bleed:true}; else delete next[item.id];return next;})}/>{t('推送', 'Send')}</Label>}</div>
          {item.item_type === 'physical' && !item.requiresShipping && <p className="text-sm text-amber-700">{t('实物规格未启用配送，请先在 Shopify 核对商品设置。', 'Shipping is disabled for this physical item. Review its Shopify settings before sending.')}</p>}
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
      {data.data.order_type !== 'digital' && <div className="flex flex-wrap gap-3"><Button disabled={busy || running || duplicate || !Object.keys(choices).length || Object.values(choices).some(c=>!c.variant_id) || !!data.data.cancelledAt || !data.data.shippingAddress} onClick={()=>void send()}>{busy || running ? t('处理中，请稍候…', 'Processing, please wait…') : t('推送到 Printful 草稿', 'Send to Printful draft')}</Button><Button variant="outline" disabled={busy} onClick={()=>void send(true)}>{t('查询 Printful 状态', 'Check Printful status')}</Button></div>}
      {data.data.order_type !== 'digital' && <p className="text-sm text-muted-foreground">{t('草稿不会自动扣款生产。请在 Printful 核对印刷文件、运费和总价后确认。', 'Drafts do not start production. Review print files, shipping and total in Printful before confirming.')}</p>}
      {result && <section className="space-y-2 rounded-xl border bg-white p-5 shadow-sm"><h3 className="flex items-center gap-2 font-medium">Printful #{result.id} <PrintfulStatus status={result.status}/></h3><p>{result.costs?.total} {result.costs?.currency}</p>{result.file_statuses?.map((f,i)=><p key={i} className="text-sm">{t('文件', 'File')} #{f.id}: {f.status === 'ok' ? t('就绪', 'Ready') : f.status === 'failed' ? t('处理失败', 'Failed') : t('处理中', 'Processing')}</p>)}{result.files?.map((f,i)=><p key={i} className="text-sm"><a className="underline" href={safeUrl(f.file_url)} target="_blank" rel="noreferrer">{t('查看印刷文件', 'View print file')} {i+1}</a> {f.dimensions?.join(' × ')}</p>)}</section>}
    </>}
  </div>;
}
