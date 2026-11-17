import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ragassist.server import Handler


@pytest.fixture(scope="module")
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def request(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_page_and_health(base_url, monkeypatch):
    status, body = request(base_url + "/")
    assert status == 200 and b"Ask the Python Docs" in body
    status, body = request(base_url + "/?q=How%20do%20I%20sort%3F&retriever=bm25")  # shareable links
    assert status == 200 and b"Ask the Python Docs" in body
    status, body = request(base_url + "/api/health")
    health = json.loads(body)
    assert status == 200 and health["chunks"] > 500 and "bm25" in health["retrievers"]


def test_ask_offline(base_url, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, body = request(base_url + "/api/ask", {"question": "How do I create a virtual environment?",
                                                   "retriever": "bm25"})
    data = json.loads(body)
    assert status == 200, data
    assert data["sources"] and data["sources"][0]["url"].startswith("https://docs.python.org/3/")
    assert "venv" in " ".join(s["source"] for s in data["sources"])


@pytest.mark.parametrize("payload,code", [
    ({"question": ""}, 400),
    ({"question": "hi", "retriever": "magic"}, 400),
])
def test_bad_requests(base_url, payload, code):
    assert request(base_url + "/api/ask", payload)[0] == code


def test_not_found(base_url):
    assert request(base_url + "/nope")[0] == 404
