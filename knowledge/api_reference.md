# Billing API Reference

The Billing API lets your backend create invoices, list payments and react to account events. All endpoints live under `https://api.example.com/v2` and accept and return JSON.

## Authentication

Every request must carry an API key in the `X-Api-Key` header. Keys are created in the dashboard under Settings → Developers and are scoped to one account.

Keys expire after 90 days. Create the new key before the old one expires, deploy it, then revoke the old key; both work during the overlap. A request with a missing, revoked or expired key gets HTTP 401.

Never put a key in a URL or in browser code. Server-side code only.

## Pagination

List endpoints return at most `page_size` items, 25 by default and 100 at most. The response includes a `next_cursor` field; pass it back as the `cursor` query parameter to get the next page. When `next_cursor` is null you have reached the last page.

Cursors stay valid for 24 hours. Results are ordered newest first and are stable while you page through them, even if new items arrive.

## Errors

Errors use standard HTTP status codes and a JSON body with `code`, `message` and `request_id`. Include the `request_id` when you contact support.

### Error codes

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | A field is missing or malformed. The `message` names the field. |
| 401 | `unauthorized` | The API key is missing, revoked or expired. |
| 403 | `forbidden` | The key is valid but belongs to another account. |
| 404 | `not_found` | No object with that id exists in this account. |
| 409 | `conflict` | The same `Idempotency-Key` was reused with a different body. |
| 429 | `quota_exceeded` | The account used its quota of 1,000 requests per hour. |
| 503 | `unavailable` | Temporary outage. Safe to retry. |

### Quotas

Quotas are counted per account, not per key, so adding keys does not raise the limit. The hourly window resets on the hour. The `X-Quota-Remaining` response header shows how many requests are left in the current window. Enterprise accounts can ask support for a higher quota.

## Idempotency

`POST` requests accept an `Idempotency-Key` header. Sending the same key and body again within 24 hours returns the original response instead of creating a second invoice. Use a random UUID per logical operation.

## Webhooks

The API sends an HTTP `POST` to your endpoint when an invoice is paid, a payment fails or a subscription renews.

### Signing

Each webhook carries an `X-Signature` header: an HMAC-SHA256 of the raw request body, keyed with your endpoint's signing secret. Compute the same HMAC and compare in constant time before trusting the payload. Reject events whose `created_at` is more than 5 minutes old to stop replays.

### Delivery and retries

Your endpoint must answer with a 2xx status within 10 seconds. Otherwise the delivery counts as failed and is retried up to 5 times with exponential backoff over 24 hours. After the last failure the endpoint is disabled and the account owner gets an email. Events can arrive more than once and out of order, so make your handler idempotent by storing each event id you've processed.
