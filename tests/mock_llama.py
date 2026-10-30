"""Stand-in for llama-server in the offline tests: /health and /v1/chat/completions answering "ok". usage: mock_llama.py PORT LOAD_DELAY_S"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, DELAY = int(sys.argv[1]), float(sys.argv[2])
KEY, START = os.environ.get('MOCK_KEY', ''), time.time()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split('?')[0] in ('/health', '/v1/health'):
            if time.time() - START < DELAY:
                return self.send_json(503, {'error': {'code': 503, 'message': 'Loading model', 'type': 'unavailable_error'}})
            return self.send_json(200, {'status': 'ok'})
        self.send_json(404, {'error': {'code': 404, 'message': 'not found', 'type': 'not_found_error'}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.headers.get('Authorization') != 'Bearer ' + KEY:
            return self.send_json(401, {'error': {'code': 401, 'message': 'Invalid API Key', 'type': 'authentication_error'}})
        if not body.get('stream'):
            return self.send_json(200, {'object': 'chat.completion', 'model': body.get('model'), 'effort': body.get('reasoning_effort'),
                                        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        for event in ({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'o'}}]},
                      {'choices': [{'index': 0, 'delta': {'content': 'k'}, 'finish_reason': 'stop'}]}):
            data = b'data: ' + json.dumps(event).encode() + b'\n\n'
            self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
        self.wfile.write(b'e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n')


ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
