import test from "node:test";
import assert from "node:assert/strict";
import { taskQuery } from "./task-query.ts";

test("Shopify launch parameters never become filters or API query parameters", () => {
  assert.equal(taskQuery("?embedded=1&shop=example.myshopify.com&host=host&hmac=signature&id_token=secret&session=session&locale=zh-CN&timestamp=123&admin_theme=admin-next"), "");
});
test("real filters and repeated conversion selections survive launch parameters", () => {
  const query = taskQuery("?host=host&user=a%40example.com&conversion_status=added_to_cart&conversion_status=checkout_completed&page=2&page_size=25&provider=");
  assert.equal(query, "?user=a%40example.com&conversion_status=added_to_cart&conversion_status=checkout_completed&page=2&page_size=25");
  assert.equal(taskQuery(query), query);
});
