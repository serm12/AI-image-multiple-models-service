import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import ArtStyleEnum, DirectoryConfig, STYLE_ALIASES, STYLE_PROMPTS
from app.core.helpers import get_style_description
from app.routers import api, admin
from app.services.style_catalog import build_art_styles_response


class StyleCatalogTests(unittest.TestCase):
    def test_catalog_and_admin_only_expose_canonical_styles(self):
        styles = build_art_styles_response()["styles"]
        values = [style["value"] for style in styles]
        self.assertEqual(len(values), 25)
        self.assertEqual(len(values), len(set(values)))
        html = admin.admin_styles(_username="test").body.decode()
        for value in values:
            self.assertIn(f"<code>{value}</code>", html)
        for legacy in STYLE_ALIASES:
            self.assertNotIn(legacy, values)
            self.assertNotIn(f"<code>{legacy}</code>", html)

    def test_all_legacy_names_resolve_to_shared_presets(self):
        for legacy, canonical in STYLE_ALIASES.items():
            with self.subTest(legacy=legacy):
                self.assertEqual(ArtStyleEnum(legacy).value, canonical)
                self.assertEqual(get_style_description(legacy), get_style_description(canonical))
                self.assertEqual(STYLE_PROMPTS[ArtStyleEnum(legacy)], STYLE_PROMPTS[canonical])
        self.assertEqual(STYLE_PROMPTS[ArtStyleEnum.realistic], "")
        with self.assertRaises(ValueError):
            ArtStyleEnum("unknown_style")

    def test_generation_saves_canonical_style_and_one_prompt_for_all_providers(self):
        app = FastAPI()
        app.include_router(api.router)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(DirectoryConfig, "TASKS_DIR", directory), \
             patch.object(api, "process_generation_background", new_callable=AsyncMock), \
             patch.object(api.task_manager, "create_task"), \
             patch.object(api.generate_request_limiter, "check", new_callable=AsyncMock) as limiter, \
             TestClient(app) as client:
            limiter.return_value.allowed = True
            for provider in ("flux_replicate", "gemini_google", "seedream-4_fal"):
                for style in ("oil_painting", "flux_oil_painting", "gemini_oil_painting", "realistic", "gemini_realistic"):
                    with self.subTest(provider=provider, style=style):
                        # Route selection is isolated from local credential configuration.
                        with patch.object(api.APIConfig, "resolve_provider", return_value=provider):
                            response = client.post("/generate-async/", data={
                                "provider": provider,
                                "prompt": "A portrait",
                                "art_style": style,
                                "input_image_url": "https://example.test/photo.png",
                            })
                        self.assertEqual(response.status_code, 200, response.text)
                        params = json.loads((Path(directory) / response.json()["task_id"] / "params.json").read_text())
                        self.assertEqual(params["art_style"], ArtStyleEnum(style).value)
                        self.assertEqual(params["prompt"], STYLE_PROMPTS[ArtStyleEnum(style)] + "A portrait")


class ProviderPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_clients_send_prepared_prompt_unchanged(self):
        from app.clients.gemini_client import GeminiClient
        from app.clients.gemini_replicate_client import GeminiReplicateClient

        for client_class, module in (
            (GeminiClient, "app.clients.gemini_client"),
            (GeminiReplicateClient, "app.clients.gemini_replicate_client"),
        ):
            client = object.__new__(client_class)
            client.model_name = "test-model"
            client.client = Mock()
            for style in ("oil_painting", "realistic", None):
                with self.subTest(client=client_class.__name__, style=style):
                    prompt = STYLE_PROMPTS.get(style, "") + "A portrait"
                    with patch(f"{module}.asyncio.to_thread", new_callable=AsyncMock,
                               side_effect=RuntimeError("request captured")) as call, \
                         patch("builtins.print"):
                        with self.assertRaisesRegex(ValueError, "request captured"):
                            await client.generate_image(prompt, art_style=style, seed=123)
                    kwargs = call.call_args.kwargs
                    actual = (kwargs["contents"][0].parts[0].text
                              if client_class is GeminiClient else kwargs["input"]["prompt"])
                    self.assertEqual(actual, prompt)


if __name__ == "__main__":
    unittest.main()
