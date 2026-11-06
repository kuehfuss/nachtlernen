import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from nachtlernen.modell import ServerModell


def test_server_modell_spricht_openai_format():
    anfragen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            daten = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            anfragen.append((self.path, daten))
            if self.path == "/v1/load_lora_adapter" and len(anfragen) > 1:
                self.send_response(400)  # schon geladen
                self.end_headers()
                return
            antwort = {"choices": [{"message": {"content": f"Antwort von {daten.get('model')}"}}]}
            body = json.dumps(antwort).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        m = ServerModell(f"http://127.0.0.1:{server.server_port}", "grund")
        m.adapter_laden("2027-07-01-v1", "/a/b")
        m.adapter_laden("2027-07-01-v1", "/a/b")  # 400 wird geschluckt
        assert m.antworten([{"role": "user", "content": "Hallo"}]) == "Antwort von grund"
        assert m.antworten([{"role": "user", "content": "Hallo"}], adapter="2027-07-01-v1") \
            == "Antwort von 2027-07-01-v1"
        assert anfragen[0] == ("/v1/load_lora_adapter", {"lora_name": "2027-07-01-v1", "lora_path": "/a/b"})
        assert anfragen[2][1]["messages"][0]["content"] == "Hallo"
    finally:
        server.shutdown()
