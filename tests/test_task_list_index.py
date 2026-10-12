import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from app.core.config import DirectoryConfig
from app.services import task_list_index as index
from app.services import task_query_service as query
from app.utils.task_utils import save_params
from app.services.storefront_events import record_storefront_event, hash_tracking_token


class TaskListIndexTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        p = patch.object(DirectoryConfig, 'TASKS_DIR', folder.name); p.start(); self.addCleanup(p.stop)
        for n in range(35):
            task = self.root / f'task-{n:03}'; task.mkdir()
            (task / 'params.json').write_text(json.dumps({'time': f'2026-10-{n % 12 + 1:02}T12:00:00+00:00',
                'client_ip': f'10.0.0.{n % 3}', 'customer_email': f'user{n}@example.com',
                'api_provider': 'fal' if n % 2 else 'provider_%', 'storefront_tracking_token_hash': hash_tracking_token('token')}))
        index.reconcile()

    def test_equivalent_pagination_and_literal_filters(self):
        for args in [dict(page=2, page_size=10), dict(page_size=10, client_ip_query='0.0.1'),
                     dict(page_size=10, user_query='USER1'), dict(provider_query='_%'),
                     dict(task_id_query='task-00'), dict(page=999, page_size=10), dict(user_query='missing')]:
            self.assertEqual(query.list_task_summaries(**args), query._list_task_summaries_from_files(**args))

    def test_warm_page_does_not_scan_task_root_or_read_other_params(self):
        reads = []
        original = query._read_json_if_exists
        def track(path):
            if str(path).endswith('params.json'): reads.append(path)
            return original(path)
        with patch.object(index, 'reconcile', side_effect=AssertionError('No full scan on warm request')), patch.object(query, '_read_json_if_exists', side_effect=track):
            result = query.list_task_summaries(page=1, page_size=10)
        self.assertEqual(len(result['tasks']), 10)
        self.assertEqual(len(reads), 10)
        self.assertEqual({Path(p).parent.name for p in reads}, {t['task_id'] for t in result['tasks']})

    def test_write_through_create_update_event_and_star(self):
        folder = self.root / 'new-task'; folder.mkdir()
        save_params({'time': '2026-10-12T14:00:00Z', 'customer_email': 'new@example.com',
                     'storefront_tracking_token_hash': hash_tracking_token('token')}, str(folder))
        self.assertEqual(query.list_task_summaries(user_query='new@')['total'], 1)
        record_storefront_event('new-task', 'token', {'event_id': 'cart', 'event_type': 'added_to_cart'})
        query.set_task_starred('new-task', True)
        self.assertEqual(query.list_task_summaries(starred_only=True, conversion_statuses=['added_to_cart'])['total'], 1)
        query.set_task_starred('new-task', False)
        self.assertEqual(query.list_task_summaries(starred_only=True)['total'], 0)

    def test_recovery_external_edit_delete_and_rebuild(self):
        folder = self.root / 'task-000'
        params = json.loads((folder / 'params.json').read_text())
        params['customer_email'] = 'changed@example.com'
        (folder / 'params.json').write_text(json.dumps(params))
        shutil.rmtree(self.root / 'task-001')
        (self.root / 'broken').mkdir(); (self.root / 'broken' / 'params.json').write_text('{')
        index.reconcile()
        self.assertEqual(query.list_task_summaries(user_query='changed@')['total'], 1)
        self.assertEqual(query.list_task_summaries()['total'], 34)
        (self.root / '.task-list-index.sqlite3').unlink()
        self.assertEqual(query.list_task_summaries()['total'], 34)

    def test_concurrent_index_updates(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(index.refresh_task, [self.root / f'task-{n:03}' for n in range(35)]))
        self.assertEqual(query.list_task_summaries()['total'], 35)
        with index.database() as db:
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
