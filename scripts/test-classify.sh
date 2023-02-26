#!/usr/bin/env bash

curl -X 'POST' \
    'http://localhost:8008/api/screennet/classify' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'message={
  "exchange_name": "string",
  "routing_key": "string",
  "message": "string",
  "queue_name": "string"
}' \
    -F 'file=@/Users/malcolm/dev/bossjones/fastapi_pytorch_postgresql_sandbox/fastapi_pytorch_postgresql_sandbox/tests/fixtures/test1.jpg;type=image/jpg'
