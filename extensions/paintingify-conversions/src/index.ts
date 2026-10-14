import {register} from '@shopify/web-pixels-extension';
import {createReporter} from './reporter.mjs';

register(({analytics, settings}) => {
  const report = createReporter({shopDomain: String(settings.shopDomain || '')});
  analytics.subscribe('checkout_started', (event) => report('checkout_started', event));
  analytics.subscribe('checkout_completed', (event) => report('checkout_completed', event));
});
