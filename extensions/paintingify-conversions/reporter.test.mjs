import test from 'node:test';
import assert from 'node:assert/strict';
import {checkoutPayloads, createReporter, ENDPOINT} from './src/reporter.mjs';
const shopDomain = 'ai-generate-image.myshopify.com';
const line = (task='task-a', id='line-a') => ({id, properties:[
  {key:'_AI Task ID',value:task},{key:'_AI Tracking Token',value:'private-tracking-token'}], variant:{id:'variant-a'}});
const event = (lines=[line()]) => ({id:'event-a',timestamp:'2026-10-14T10:00:00Z',clientId:'visitor',
  data:{checkout:{token:'checkout-a',lineItems:lines}}});
test('maps duplicate task lines once, skips untracked items, tolerates redacted email', () => {
  const payloads = checkoutPayloads('checkout_started', event([line(),line(),{id:'ordinary',properties:[]}]),shopDomain);
  assert.equal(payloads.length,1);
  assert.equal(payloads[0].customer_email,'');
  assert.equal(payloads[0].event_id,'checkout_started:event-a:line-a');
  assert.equal(payloads[0].tracking_token,'private-tracking-token');
  assert.equal(payloads[0].source,'shopify_app_pixel');
  assert.deepEqual(checkoutPayloads('checkout_started',{},shopDomain),[]);
  assert.deepEqual(checkoutPayloads('page_viewed',event(),shopDomain),[]);
});
test('retries transient errors with identical IDs and ignores replay', async () => {
  const calls=[];
  const report=createReporter({shopDomain,wait:async()=>{},send:async(url,options)=>{
    calls.push({url,...options});
    return {ok:calls.length===3,status:calls.length===1?429:503};
  }});
  await report('checkout_started',event());
  await report('checkout_started',event());
  assert.equal(calls.length,3);
  assert.equal(calls[0].url,ENDPOINT);
  assert.equal(calls[0].credentials,'omit');
  assert.equal(calls[0].body,calls[2].body);
  assert.equal(calls[0].headers['Content-Type'],'text/plain;charset=UTF-8');
});
test('permanent 403 is not retried; other tasks still send', async () => {
  let calls=0;
  const report=createReporter({shopDomain,send:async()=>{calls++;return {ok:false,status:403};}});
  await report('checkout_completed',event([line(),line('task-b')]));
  assert.equal(calls,2);
});
test('network failure is retried and concurrent replay shares one delivery', async () => {
  let calls=0;
  const report=createReporter({shopDomain,wait:async()=>{},send:async()=>{
    calls++;if(calls===1)throw Error('network');return {ok:true,status:200};
  }});
  await Promise.all([report('checkout_started',event()),report('checkout_started',event())]);
  assert.equal(calls,2);
});
test('distinct stages both send and invalid settings never send', async () => {
  let calls=0;const send=async()=>{calls++;return {ok:true,status:200};};
  const report=createReporter({shopDomain,send});
  await report('checkout_started',event());await report('checkout_completed',event());
  assert.equal(calls,2);
  await createReporter({shopDomain:'https://attacker.example',send})('checkout_started',event());
  assert.equal(calls,2);
});
