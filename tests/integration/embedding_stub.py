import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DIMENSIONS = 16


def vector(value):
    raw = hashlib.sha256(str(value).encode()).digest()
    return [round(raw[index % len(raw)] / 255, 6) for index in range(DIMENSIONS)]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"data": [{"id": "deterministic-test-model", "object": "model"}]}
        self.respond(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        inputs = request["input"]
        inputs = inputs if isinstance(inputs, list) else [inputs]
        self.respond({"data": [{"index": index, "embedding": vector(text)}
                               for index, text in enumerate(inputs)]})

    def respond(self, body):
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


ThreadingHTTPServer(("0.0.0.0", 8081), Handler).serve_forever()
