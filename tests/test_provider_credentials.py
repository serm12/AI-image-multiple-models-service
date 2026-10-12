import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.core.config import DirectoryConfig, APIConfig
from app.services import provider_credentials as settings
from app.routers.provider_credentials import router
from app.services.security import require_admin_login


class ProviderCredentialsTests(unittest.TestCase):
    def test_encrypted_persistence_no_disclosure_and_blank_preserves(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(DirectoryConfig,'TASKS_DIR',folder), patch.dict(os.environ,{'SHOPIFY_API_SECRET':'test-encryption-secret'}):
            app=FastAPI();app.include_router(router)
            with TestClient(app,base_url='https://image-api.shopupup.com') as client:
                self.assertNotEqual(client.get('/admin/provider-credentials').status_code,200)
                app.dependency_overrides[require_admin_login]=lambda:'admin'
                values={'AIAPIROUTE_API_KEY':'fake-private-aiapiroute','FAL_API_KEY':'fake-private-fal'}
                self.assertEqual(client.post('/admin/provider-credentials',json=values).status_code,403)
                headers={'Origin':'https://image-api.shopupup.com'}
                result=client.post('/admin/provider-credentials',json=values,headers=headers)
                self.assertEqual(result.status_code,200)
                for value in values.values():
                    self.assertNotIn(value,result.text)
                    self.assertNotIn(value.encode(),settings.path().read_bytes())
                    self.assertNotIn(value,client.get('/admin/provider-credentials').text)
                self.assertEqual(APIConfig.AIAPIROUTE_API_KEY,values['AIAPIROUTE_API_KEY'])
                self.assertEqual(APIConfig.FAL_API_KEY,values['FAL_API_KEY'])
                client.post('/admin/provider-credentials',json={'FAL_API_KEY':''},headers=headers)
                self.assertEqual(APIConfig.FAL_API_KEY,values['FAL_API_KEY'])
                result=client.post('/admin/provider-credentials',json={'FAL_API_KEY':['secret-do-not-echo']},headers=headers)
                self.assertEqual(result.status_code,400)
                self.assertNotIn('secret-do-not-echo',result.text)
                from app.clients.fal_gpt_image2_client import FalGPTImage2Client
                from app.clients.unified_api_client import UnifiedAPIClient
                unified = UnifiedAPIClient()
                first=unified._get_fal_gpt_image2_client()
                old_ai = unified._get_aiapiroute_client("test-model")
                settings.save({'FAL_API_KEY':'replacement-test-key','AIAPIROUTE_API_KEY':'replacement-ai-key'})
                second=unified._get_fal_gpt_image2_client()
                self.assertEqual(unified._get_aiapiroute_client("test-model").api_key,"replacement-ai-key")
                self.assertEqual(old_ai.api_key,values["AIAPIROUTE_API_KEY"])
                self.assertEqual(first.fal.key,'fake-private-fal')
                self.assertEqual(second.fal.key,'replacement-test-key')
                self.assertEqual(settings.resolve('AIAPIROUTE_API_KEY'),'replacement-ai-key')
