"""The production React routes: permissions, real task data, and build delivery."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.core.config import AppConfig, DirectoryConfig
from app.services.time_settings import update
from app.services.i18n import save_locale
from app.utils.time_utils import now_china, CHINA_TIMEZONE_NAME

class AdminFrontendTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for obj, name, value in [(DirectoryConfig,'TASKS_DIR',self.folder.name),(AppConfig,'ADMIN_USER','admin'),(AppConfig,'ADMIN_PASSWORD','test-secret')]:
            p=patch.object(obj,name,value);p.start();self.addCleanup(p.stop)
        update({'shop_timezone':'Asia/Shanghai'})
        self.client=TestClient(app,base_url='https://testserver')
        self.client.auth=('admin','test-secret')
        self.addCleanup(self.client.close)
        self.origin={'Origin':'https://testserver'}

    def task(self, name, **values):
        folder=Path(self.folder.name,name);folder.mkdir()
        params={'time':now_china().isoformat(),'time_zone':CHINA_TIMEZONE_NAME,'client_ip':'203.0.113.1','api_provider':'fal','customer_email':'buyer@example.com','original_prompt':'<script>alert(1)</script>',**values}
        (folder/'params.json').write_text(json.dumps(params),'utf-8')
        (folder/'bfl_response.json').write_text(json.dumps({'status':'completed'}),'utf-8')
        Image.new('RGB',(40,40),'blue').save(folder/'result.png')
        return folder

    def test_all_pages_require_auth_and_deliver_react_shell_and_assets(self):
        for path in ['/admin/tasks','/admin/styles','/admin/config','/admin/api/session','/admin/api/tasks','/admin/api/styles']:
            denied=self.client.get(path,auth=None,follow_redirects=False)
            self.assertEqual(denied.status_code,303,path)
            self.assertTrue(denied.headers['location'].startswith('/admin/login'))
        for path in ['/admin/tasks','/admin/styles','/admin/config']:
            response=self.client.get(path)
            self.assertEqual(response.status_code,200)
            self.assertIn('<div id="root"></div>',response.text)
            self.assertEqual(response.headers['cache-control'],'no-store')
            for asset in re.findall(r'(?:src|href)="(/admin/assets/[^"]+)"',response.text):
                self.assertEqual(self.client.get(asset,auth=None).status_code,200,asset)
        self.assertEqual(self.client.get('/admin/assets/../../.env',auth=None).status_code,404)

    def test_login_error_remember_cookie_and_logout(self):
        client=TestClient(app,base_url='https://testserver')
        self.addCleanup(client.close)
        bad=client.post('/admin/login',data={'username':'admin','password':'bad'})
        self.assertEqual(bad.status_code,401)
        self.assertIn('id="root"',bad.text)
        response=client.post('/admin/login',data={'username':'admin','password':'test-secret','remember':'on','next':'https://evil.example'},follow_redirects=False)
        self.assertEqual(response.status_code,303)
        self.assertEqual(response.headers['location'],'/admin/tasks')
        cookie=response.headers['set-cookie']
        for expected in ['HttpOnly','Secure','SameSite=strict','Max-Age=']:
            self.assertIn(expected,cookie)
        self.assertEqual(client.get('/admin/api/session').json()['username'],'admin')
        client.post('/admin/logout')
        self.assertEqual(client.get('/admin/api/tasks',follow_redirects=False).status_code,303)

    def test_task_data_filters_pagination_references_and_thumbnail(self):
        self.task('task-1',api_provider='gemini',customer_email='other@example.com')
        self.task('task-2',source_task_id='task-1',edit_instructions='Make it blue')
        for index in range(10):
            self.task(f'extra-{index}',api_provider='other',customer_email='extra@example.com',client_ip='203.0.113.2')
        data=self.client.get('/admin/api/tasks?page_size=10').json()
        self.assertEqual(data['total'],12)
        self.assertEqual(len(data['tasks']),10)
        self.assertTrue(data['has_next'])
        for query in ['task_id=task-2','provider=fal','user=buyer@example.com','client_ip=203.0.113.1&provider=fal','time_range=today&provider=fal']:
            result=self.client.get('/admin/api/tasks?'+query)
            self.assertEqual(result.status_code,200)
            self.assertEqual([t['task_id'] for t in result.json()['tasks']],['task-2'],query)
        item=self.client.get('/admin/api/tasks?task_id=task-2').json()['tasks'][0]
        self.assertEqual(item['prompt'],'<script>alert(1)</script>')
        self.assertEqual(item['edit_instructions'],'Make it blue')
        self.assertIn('Asia/Shanghai',item['display_times'])
        self.assertEqual(item['ip_task_total'],1)
        self.assertEqual(item['ip_task_sequence'],1)
        image=item['images'][-1]
        thumbnail=self.client.get(image['thumbnail'])
        self.assertEqual(thumbnail.status_code,200)
        self.assertEqual(thumbnail.headers['content-type'],'image/jpeg')
        self.assertNotIn('<script>alert(1)</script>',self.client.get('/admin/tasks').text)
        self.assertEqual(self.client.get('/admin/api/tasks?page_size=1000').status_code,422)

    def test_star_explicit_state_origin_and_filter(self):
        self.task('task-1');self.task('task-2')
        path='/admin/api/tasks/task-1/star'
        self.assertEqual(self.client.post(path,json={'starred':True}).status_code,403)
        self.assertEqual(self.client.post(path,json={'starred':True},headers={'Origin':'https://evil.example'}).status_code,403)
        for _ in range(2):
            self.assertEqual(self.client.post(path,json={'starred':True},headers=self.origin).json(),{'starred':True})
        self.assertEqual([t['task_id'] for t in self.client.get('/admin/api/tasks?starred=true').json()['tasks']],['task-1'])
        self.client.post(path,json={'starred':False},headers=self.origin)
        self.assertEqual(self.client.get('/admin/api/tasks?starred=true').json()['total'],0)
        self.assertEqual(self.client.post('/admin/api/tasks/missing/star',json={'starred':True},headers=self.origin).status_code,404)

    def test_styles_and_language_session(self):
        styles=self.client.get('/admin/api/styles').json()
        self.assertGreater(len(styles),0)
        self.assertIn('prompt',styles[0])
        for locale in ['en','zh-CN']:
            save_locale(locale)
            response=self.client.get('/admin/config')
            self.assertIn('lang="'+locale+'"',response.text)
            session=self.client.get('/admin/api/session').json()
            self.assertEqual(session['locale'],locale)
            self.assertIn('version',session)
            self.assertNotIn('password',json.dumps(session))

if __name__ == '__main__': unittest.main()
