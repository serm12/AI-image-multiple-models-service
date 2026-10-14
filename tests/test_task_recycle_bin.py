import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import DirectoryConfig, AppConfig
from app.services import task_recycle_bin as recycle
from app.services import task_list_index as index
from app.services.task_query_service import list_task_summaries, _list_task_summaries_from_files, set_task_starred
from app.services.async_task_manager import task_manager
from app.services.r2_storage import R2StorageConfig, delete_task_images


class RecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for obj, key, value in [(DirectoryConfig,'TASKS_DIR',self.temp.name),(AppConfig,'ADMIN_USER','admin'),(AppConfig,'ADMIN_PASSWORD','test-secret')]:
            p=patch.object(obj,key,value);p.start();self.addCleanup(p.stop)
        self.now = datetime(2026,10,14,12,tzinfo=timezone.utc)
        self.client=TestClient(app,base_url='https://testserver');self.client.auth=('admin','test-secret')
        self.addCleanup(self.client.close)
        self.origin={'Origin':'https://testserver'}
        self.make('task-a');self.make('task-b')

    def make(self, task_id):
        path=self.root/task_id;path.mkdir()
        (path/'params.json').write_text(json.dumps({'time':self.now.isoformat(),'client_ip':'1.2.3.4'}))
        (path/'result.png').write_bytes(b'image-data')
        return path

    def test_trash_restore_preserve_files_star_and_rebuild(self):
        set_task_starred('task-a',True)
        self.assertEqual(recycle.change_tasks(['task-a','task-a'],now=self.now)['deleted'],['task-a'])
        self.assertEqual(list_task_summaries()['total'],1)
        trashed=list_task_summaries(recycle_bin=True)
        self.assertEqual(trashed['total'],1)
        self.assertTrue(trashed['tasks'][0]['starred'])
        self.assertEqual(trashed['tasks'][0]['purge_after'],(self.now+timedelta(days=30)).isoformat())
        self.assertEqual((self.root/'task-a/result.png').read_bytes(),b'image-data')
        recycle.change_tasks(['task-a'],now=self.now+timedelta(days=1))
        self.assertEqual(recycle.metadata('task-a')['trashed_at'],self.now.isoformat())
        (self.root/'.task-list-index.sqlite3').unlink()
        self.assertEqual(list_task_summaries(recycle_bin=True)['total'],1)
        self.assertEqual(list_task_summaries(recycle_bin=True),_list_task_summaries_from_files(recycle_bin=True))
        recycle.change_tasks(['task-a'],restore=True)
        self.assertEqual(list_task_summaries()['total'],2)
        self.assertEqual(list_task_summaries(recycle_bin=True)['total'],0)
        self.assertTrue(list_task_summaries(task_id_query='task-a')['tasks'][0]['starred'])

    def test_api_requires_auth_origin_confirmation_and_bounded_safe_ids(self):
        payload={'task_ids':['task-a'],'confirmed':True}
        denied=self.client.post('/admin/api/tasks/delete',json=payload,auth=None,headers=self.origin,follow_redirects=False)
        self.assertEqual(denied.status_code,303)
        self.assertEqual(self.client.post('/admin/api/tasks/delete',json=payload).status_code,403)
        self.assertEqual(self.client.post('/admin/api/tasks/delete',json=payload,headers={'Origin':'https://evil.example'}).status_code,403)
        for value in [{'task_ids':['task-a']},{'task_ids':['task-a'],'confirmed':False},
                      {'task_ids':['../task-a'],'confirmed':True},{'task_ids':[],'confirmed':True},
                      {'task_ids':['task-a']*101,'confirmed':True}]:
            self.assertEqual(self.client.post('/admin/api/tasks/delete',json=value,headers=self.origin).status_code,422)
        self.assertFalse(recycle.metadata('task-a'))
        self.assertEqual(self.client.post('/admin/api/tasks/delete',json=payload,headers=self.origin).json()['deleted'],['task-a'])
        self.assertEqual(self.client.get('/admin/api/tasks?recycle_bin=true').json()['total'],1)
        self.assertEqual(self.client.post('/admin/api/tasks/restore',json=payload,headers=self.origin).json()['deleted'],['task-a'])

    def test_running_tasks_and_partial_batch_failures(self):
        task_manager.tasks['task-a']={'status':'generating'}
        self.addCleanup(task_manager.tasks.pop,'task-a',None)
        result=recycle.change_tasks(['task-a','task-b','missing'])
        self.assertEqual(result['deleted'],['task-b'])
        self.assertEqual([x['reason'] for x in result['failed']],['task_running','task_not_found'])
        self.assertTrue((self.root/'task-a/params.json').exists())

    def test_only_expired_tasks_purge_and_restored_tasks_survive(self):
        recycle.change_tasks(['task-a','task-b'],now=self.now)
        recycle.change_tasks(['task-b'],restore=True)
        delete_cloud=Mock()
        self.assertEqual(recycle.purge_expired(self.now+timedelta(days=30,seconds=-1),delete_cloud=delete_cloud),0)
        delete_cloud.assert_not_called()
        self.assertEqual(recycle.purge_expired(self.now+timedelta(days=30),delete_cloud=delete_cloud),1)
        delete_cloud.assert_called_once_with('task-a',self.root/'task-a')
        self.assertFalse((self.root/'task-a').exists())
        self.assertTrue((self.root/'task-b/result.png').exists())
        self.assertEqual(list_task_summaries()['total'],1)
        self.assertEqual(recycle.purge_expired(self.now+timedelta(days=31),delete_cloud=delete_cloud),0)

    def test_cloud_failure_is_retryable_and_cannot_restore_partial_cleanup(self):
        recycle.change_tasks(['task-a'],now=self.now)
        cloud=Mock(side_effect=RuntimeError('cloud_failed'))
        due=self.now+timedelta(days=30)
        self.assertEqual(recycle.purge_expired(due,delete_cloud=cloud),0)
        self.assertTrue((self.root/'task-a/result.png').exists())
        self.assertTrue(recycle.metadata('task-a')['cleanup_started_at'])
        self.assertEqual(recycle.change_tasks(['task-a'],restore=True)['failed'][0]['reason'],'cleanup_in_progress')
        self.assertEqual(recycle.purge_expired(due,delete_cloud=Mock()),1)
        self.assertFalse(recycle.marker_path('task-a').exists())

    def test_partial_local_failure_retains_external_marker_for_retry(self):
        recycle.change_tasks(['task-a'],now=self.now)
        due=self.now+timedelta(days=30)
        with patch.object(recycle.shutil,'rmtree',side_effect=OSError('disk_busy')):
            self.assertEqual(recycle.purge_expired(due,delete_cloud=Mock()),0)
        self.assertTrue(recycle.marker_path('task-a').exists())
        self.assertEqual(recycle.purge_expired(due,delete_cloud=Mock()),1)

    def test_rejects_symlink_to_another_task(self):
        try: (self.root/'linked').symlink_to(self.root/'task-a',target_is_directory=True)
        except OSError: self.skipTest('symlink privileges unavailable')
        self.assertEqual(recycle.change_tasks(['linked'])['failed'][0]['reason'],'unsafe_task_path')
        self.assertTrue((self.root/'task-a/result.png').exists())

    def test_last_page_clamps_after_trash(self):
        for n in range(10):self.make('extra-'+str(n))
        last=list_task_summaries(page=2,page_size=10)
        recycle.change_tasks([t['task_id'] for t in last['tasks']])
        data=list_task_summaries(page=2,page_size=10)
        self.assertEqual(data['page'],1)
        self.assertEqual(data['total'],10)


class CloudDeleteTests(unittest.TestCase):
    def config(self):
        return R2StorageConfig('account','key','secret','bucket','https://cdn.example')

    def test_deletes_exact_task_prefix_and_all_pages(self):
        client=Mock()
        client.get_paginator.return_value.paginate.return_value=[{'Contents':[{'Key':'taskfile/task-a/result.png'}]},
            {'Contents':[{'Key':'taskfile/task-a/preview.png'}]}]
        client.delete_objects.return_value={}
        delete_task_images('task-a','unused',config=self.config(),s3_client=client)
        client.get_paginator.return_value.paginate.assert_called_once_with(Bucket='bucket',Prefix='taskfile/task-a/')
        self.assertEqual(client.delete_objects.call_count,2)

    def test_rejects_foreign_keys_and_partial_cloud_errors(self):
        client=Mock();client.get_paginator.return_value.paginate.return_value=[{'Contents':[{'Key':'taskfile/task-ab/result.png'}]}]
        with self.assertRaises(RuntimeError):delete_task_images('task-a','unused',config=self.config(),s3_client=client)
        client.delete_objects.assert_not_called()
        client.get_paginator.return_value.paginate.return_value=[{'Contents':[{'Key':'taskfile/task-a/result.png'}]}]
        client.delete_objects.return_value={'Errors':[{'Code':'AccessDenied'}]}
        with self.assertRaises(RuntimeError):delete_task_images('task-a','unused',config=self.config(),s3_client=client)
        with self.assertRaises(ValueError):delete_task_images('../other','unused',config=self.config(),s3_client=client)
