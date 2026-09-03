# Provider Fallbacks

The image generation API can try multiple providers for the same model family.
If the first provider fails, the service automatically tries the next configured provider.

Default groups:

```json
{
  "flux": ["flux_bfl", "flux_replicate", "flux_fireworks"],
  "gpt-image-2": ["gpt-image-2_aiapiroute", "gpt-image-2_fal"],
  "seedream-4": ["seedream-4_replicate", "seedream-4_fal"],
  "gemini-nanobanana": [
    "gemini-nanobanana_google",
    "gemini-nanobanana_replicate",
    "gemini-nanobanana_openrouter"
  ]
}
```

To change the order or add a group, set `PROVIDER_FALLBACK_GROUPS_JSON` in `.env`:

```env
PROVIDER_FALLBACKS_ENABLED=true
PROVIDER_FALLBACK_GROUPS_JSON={"flux":["flux_bfl","flux_replicate","flux_fireworks"],"gpt-image-2":["gpt-image-2_aiapiroute","gpt-image-2_fal"]}
```

Notes:

- You can still pass a specific provider such as `flux_replicate`; the chain starts from that provider and then tries the remaining providers in the same group.
- You can pass a group alias such as `flux` or `gpt-image-2`; the chain starts from the first provider in that group.
- Providers without required API keys are skipped when building the chain.
- Each fallback provider still needs to support the requested `aspect_ratio`.
- `params.json` records `api_provider`, `requested_api_provider`, `provider_fallback_chain`, and `provider_fallback_attempts`.
