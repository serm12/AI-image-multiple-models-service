import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from app.services import shopify_customers as service


def customer(state='SUBSCRIBED', email='person@example.com'):
    return {'id': 'gid://shopify/Customer/1', 'email': email,
            'emailMarketingConsent': {'marketingState': state}}


class CustomerSyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        p = patch.object(service.DirectoryConfig, 'TASKS_DIR', folder.name)
        p.start(); self.addCleanup(p.stop)
        self.task_id = '20261012_120000_1234abcd'
        path = self.root / self.task_id
        path.mkdir()
        (path / 'params.json').write_text(json.dumps({'customer_email': 'person@example.com'}))
        self.event = {'event_type': 'newsletter_result', 'source': 'theme_newsletter_response',
            'subscription_status': 'failed', 'customer_email': 'person@example.com',
            'recorded_at': '2026-10-12T12:00:00+00:00'}

    def test_enqueue_rejects_email_switch_and_non_signup(self):
        self.assertFalse(service.enqueue(self.task_id, {**self.event, 'customer_email': 'other@example.com'}))
        self.assertFalse(service.enqueue(self.task_id, {**self.event, 'subscription_status': 'not_attempted'}))
        self.assertFalse(service.enqueue('../private', self.event))
        self.assertTrue(service.enqueue(self.task_id, self.event))
        self.assertFalse(service.enqueue(self.task_id, self.event))
        self.assertEqual(len(service.summaries()), 1)

    async def test_create_includes_actual_single_opt_in(self):
        execute = AsyncMock(side_effect=[{'customers': {'nodes': []}},
            {'customerCreate': {'customer': customer(), 'userErrors': []}}])
        result = await service.synchronize('person@example.com', self.event['recorded_at'], execute)
        self.assertEqual(result[0], 'subscribed')
        value = execute.call_args.args[1]['input']
        self.assertEqual(value['emailMarketingConsent']['marketingOptInLevel'], 'SINGLE_OPT_IN')
        self.assertNotIn('verifiedEmail', value)

    async def test_subscribed_and_optout_are_not_mutated(self):
        for state in ('SUBSCRIBED', 'UNSUBSCRIBED', 'PENDING', 'INVALID'):
            execute = AsyncMock(return_value={'customers': {'nodes': [customer(state)]}})
            result = await service.synchronize('person@example.com', self.event['recorded_at'], execute)
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(result[0], 'subscribed' if state == 'SUBSCRIBED' else 'preserved')

    async def test_existing_customer_subscription(self):
        execute = AsyncMock(side_effect=[{'customers': {'nodes': [customer('NOT_SUBSCRIBED')]}},
            {'customerEmailMarketingConsentUpdate': {'customer': customer(), 'userErrors': []}}])
        result = await service.synchronize('person@example.com', self.event['recorded_at'], execute)
        self.assertEqual(result[0], 'subscribed')
        self.assertEqual(execute.call_args.args[0], service.SUBSCRIBE)

    async def test_create_race_requeries_exact_email(self):
        execute = AsyncMock(side_effect=[{'customers': {'nodes': [customer(email='wrong@example.com')]}},
            {'customerCreate': {'customer': None, 'userErrors': [{'message': 'taken'}]}},
            {'customers': {'nodes': [customer()]}}])
        self.assertEqual((await service.synchronize('person@example.com', self.event['recorded_at'], execute))[0], 'subscribed')

    async def test_failure_persists_retry_then_success(self):
        service.enqueue(self.task_id, self.event)
        with patch.object(service, 'synchronize', AsyncMock(side_effect=RuntimeError('temporary'))):
            await service.process_pending()
        self.assertEqual(service.summaries()[self.task_id]['state'], 'retry')
        with service.database() as db:
            db.execute('UPDATE jobs SET retry_at=0')
        with patch.object(service, 'synchronize', AsyncMock(return_value=('subscribed', 'customer-1', 'ok'))):
            await service.process_pending()
        self.assertEqual(service.summaries()[self.task_id]['state'], 'subscribed')

    def test_token_encrypted_and_bound_to_shop(self):
        with patch.dict('os.environ', {'SHOPIFY_CLIENT_SECRET': 'test-only-secret', 'SHOPIFY_API_SECRET': ''}), patch.object(service, 'allowed_shop', return_value='test-shop'):
            encrypted = service.cipher().encrypt(json.dumps({'shop': 'test-shop', 'token': 'private-token'}).encode()).decode()
            with service.database() as db:
                db.execute("INSERT INTO config VALUES ('token', ?)", (encrypted,))
            self.assertEqual(service.saved_token(), 'private-token')
            self.assertNotIn(b'private-token', (self.root / '.shopify-customer-sync.sqlite3').read_bytes())
            service.clear_token()
            self.assertIsNone(service.saved_token())
