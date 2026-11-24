import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from ml.datasets import parallel_resume as pr


@pytest.mark.parametrize("start, end, parts", [(0, 10, 3), (5, 6, 4), (100, 1000, 8), (0, 7, 7)])
def test_split_ranges_cover_exactly_once(start, end, parts):
    ranges = pr.split_ranges(start, end, parts)
    covered = [b for first, last in ranges for b in range(first, last + 1)]
    assert covered == list(range(start, end)) and all(first <= last for first, last in ranges)
    assert len(ranges) == min(parts, end - start)


def test_split_ranges_empty_when_complete():
    assert pr.split_ranges(10, 10, 4) == [] and pr.split_ranges(11, 10, 4) == []


class RangeHandler(SimpleHTTPRequestHandler):
    """Static server with byte-range support and a scripted failure, standing in for Cloud Storage."""

    fail_once: ClassVar[set[str]] = set()

    def do_GET(self):
        with open(self.directory + "/blob", "rb") as handle:
            payload = handle.read()
        first, last = self.headers["Range"].removeprefix("bytes=").split("-")
        first, last = int(first), int(last)
        key = self.headers["Range"]
        body = payload[first : last + 1]
        if key in RangeHandler.fail_once:
            RangeHandler.fail_once.discard(key)
            body = body[: len(body) // 2]  # truncated response: the client must resume the rest
            self.send_response(206)
            self.send_header("Content-Length", str(last - first + 1))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.connection.close()
            return
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(tmp_path):
    blob = bytes(range(256)) * 4000 + b"tail"
    (tmp_path / "blob").write_bytes(blob)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/blob", blob
    httpd.shutdown()


def test_finish_resumes_partial_file_in_parallel_and_matches_md5(tmp_path, server, monkeypatch):
    url, blob = server
    monkeypatch.setattr(pr.time, "sleep", lambda s: None)
    target = tmp_path / "out.zip"
    target.write_bytes(blob[:123_457])  # what the throttled single connection already fetched
    RangeHandler.fail_once = {f"bytes={123_457}-{123_457 + (len(blob) - 123_457) // 4 - 1}"}
    pr.finish(target, url, len(blob), connections=4)
    assert target.read_bytes() == blob
    assert pr.md5_of(target) == hashlib.md5(blob).hexdigest()
    assert not list(tmp_path.glob("out.zip.part*"))


def test_finish_is_a_no_op_when_already_complete(tmp_path, server):
    url, blob = server
    target = tmp_path / "done.zip"
    target.write_bytes(blob)
    pr.finish(target, url, len(blob), connections=4)
    assert target.read_bytes() == blob


def test_size_mismatch_is_reported(tmp_path, server):
    url, blob = server
    target = tmp_path / "x.zip"
    target.write_bytes(blob[:10])
    with pytest.raises(SystemExit, match="server has"):
        pr.finish(target, url, len(blob) + 5, connections=2)
    assert target.read_bytes() == blob[:10]  # nothing appended when the size is wrong


def test_remote_size_reads_content_range(server):
    url, blob = server
    assert pr.remote_size(url) == len(blob)


def test_complete_part_makes_no_request(tmp_path, monkeypatch):
    part = tmp_path / "p"
    part.write_bytes(b"x" * 10)

    def no_network(*args, **kwargs):
        raise AssertionError("a complete part must not be requested again")

    monkeypatch.setattr(pr.urllib.request, "urlopen", no_network)
    assert pr.fetch_range("http://unused", 100, 109, part) == part
