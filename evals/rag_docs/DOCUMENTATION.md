# Project Documentation

## Vector Store

Chunks are stored in ChromaDB. Call `get_or_create_collection` to open the collection without wiping existing data.

## Rate Limits

The API allows 60 requests per minute per key. Exceeding it returns HTTP 429; wait and retry with exponential backoff.

## Deployment

Build the Docker image with `docker build -t docs-assistant .` and run it with port 8080 exposed.
