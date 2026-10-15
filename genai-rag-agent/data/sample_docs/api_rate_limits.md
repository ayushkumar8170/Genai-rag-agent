# API Rate Limits

The NimbusDesk API limits how many requests each workspace can send per minute. The Starter plan allows 60 requests per minute, the Business plan allows 600 requests per minute, and Enterprise limits are agreed per contract.

## What happens when you exceed the limit

Requests over the limit receive an HTTP 429 Too Many Requests response with a Retry-After header that says how many seconds to wait. We recommend exponential backoff with jitter. Short bursts of up to twice your limit are tolerated for 10 seconds before throttling begins.

## Bulk endpoints

Bulk endpoints accept at most 100 items per request. Larger jobs should be split into several requests or run through the asynchronous import endpoint.

## API keys

Create keys under Settings, then Developers. Keys never expire automatically, but we recommend rotating them every 90 days and storing them in a secrets manager rather than in source code.
