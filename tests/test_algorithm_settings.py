import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.core.config import AlgorithmConfig, DirectoryConfig
from app.services import algorithm_settings as settings
from app.routers.algorithm_settings import router
from app.services.security import require_admin_login
from app.utils.task_utils import generate_output_filenames


class AlgorithmSettingsTests(unittest.TestCase):
    def test_runtime_factor_persistence_and_validation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(DirectoryConfig, 'TASKS_DIR', folder), patch.object(AlgorithmConfig, 'ALGORITHM_FACTOR', 6987):
            app=FastAPI();app.include_router(router)
            with TestClient(app, base_url='https://image-api.shopupup.com') as client:
                self.assertNotEqual(client.get('/admin/algorithm-settings').status_code, 200)
                app.dependency_overrides[require_admin_login]=lambda:'admin'
                self.assertEqual(client.get('/admin/algorithm-settings').json()['factor'], 6987)
                self.assertEqual(client.post('/admin/algorithm-settings', json={'factor':1234}).status_code, 403)
                headers={'Origin':'https://image-api.shopupup.com'}
                for invalid in [0,-1,1.5,True,1000000000,'1234']:
                    self.assertEqual(client.post('/admin/algorithm-settings',json={'factor':invalid},headers=headers).status_code,422)
                original=Path(folder)/'output_cropped_original_legacy.png';original.write_bytes(b'preserve')
                self.assertEqual(client.post('/admin/algorithm-settings',json={'factor':1234},headers=headers).status_code,200)
                self.assertEqual(settings.factor(),1234)
                expected=datetime.now().year*1234
                self.assertEqual(AlgorithmConfig.get_algorithm_value(),expected)
                self.assertIn(str(expected),generate_output_filenames(folder,'input.png','png')[1])
                self.assertEqual(original.read_bytes(),b'preserve')
                settings.path().write_text('{')
                self.assertEqual(settings.factor(),6987)
