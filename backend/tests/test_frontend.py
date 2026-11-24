import json
import os

import pytest

from backend.app.main import create_app
from backend.tests.helpers import assert_error, call, make_settings

INDEX = b"<!doctype html><title>spa</title>"
SCRIPT = b"console.log('app')"
SECRET = b"TOP-SECRET-OUTSIDE-DIST"


@pytest.fixture
def site(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_bytes(INDEX)
    (dist / "assets" / "app.js").write_bytes(SCRIPT)
    (tmp_path / "secret.txt").write_bytes(SECRET)
    os.symlink(tmp_path / "secret.txt", dist / "leak.txt")
    os.symlink(tmp_path, dist / "escape")
    return dist


@pytest.fixture
def spa_client(make_client, site):
    return make_client(frontend_dist=site)


def test_real_files_are_served(spa_client):
    response = spa_client.get("/assets/app.js")
    assert response.status_code == 200 and response.content == SCRIPT
    assert "javascript" in response.headers["content-type"]
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("path", ["/", "/about", "/results/42", "/deep/client/route/"])
def test_client_routes_fall_back_to_index(spa_client, path):
    response = spa_client.get(path)
    assert response.status_code == 200 and response.content == INDEX
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_head_requests_are_served_without_body(spa_client):
    response = spa_client.head("/about")
    assert response.status_code == 200 and response.content == b""
    assert response.headers["content-length"] == str(len(INDEX))


def test_missing_asset_is_404_not_index(spa_client):
    assert_error(spa_client.get("/assets/missing.js"), 404, "not_found")
    assert_error(spa_client.get("/favicon.ico"), 404, "not_found")


@pytest.mark.parametrize("path", ["/api", "/api/", "/api/unknown", "/api/predict/extra", "/api/health.json"])
def test_unknown_api_paths_are_json_404(spa_client, path):
    response = spa_client.get(path)
    assert_error(response, 404, "not_found")
    assert INDEX not in response.content


def test_wrong_method_on_known_api_route_stays_405(spa_client):
    assert_error(spa_client.get("/api/predict"), 405, "method_not_allowed")


def test_non_get_on_client_route_is_404(spa_client):
    assert_error(spa_client.post("/about"), 404, "not_found")
    assert_error(spa_client.post("/api/unknown"), 404, "not_found")


def test_symlinks_out_of_dist_are_refused(spa_client):
    assert_error(spa_client.get("/leak.txt"), 404, "not_found")
    assert_error(spa_client.get("/escape/secret.txt"), 404, "not_found")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw_path",
    [
        "/../secret.txt",
        "/assets/../../secret.txt",
        "//..//secret.txt",
        "/assets/%2e%2e/../../secret.txt",
        "/..%2fsecret.txt",
        "/" + "../" * 20 + "etc/hosts.txt",
        "/escape/../../secret.txt",
        "/\x00secret.txt",
    ],
)
async def test_path_traversal_is_impossible(tmp_path, site, raw_path):
    """Raw ASGI paths, because HTTP clients normalise `..` away before sending."""
    exchange = await call(create_app(make_settings(tmp_path, frontend_dist=site)), [b""], path=raw_path, method="GET")
    assert exchange.status == 404 and json.loads(exchange.body)["error"]["code"] == "not_found"
    assert SECRET not in exchange.body


@pytest.mark.anyio
@pytest.mark.parametrize("raw_path", ["/../outside", "/assets/../../outside", "/" + "../" * 20 + "etc/hosts"])
async def test_traversal_without_suffix_falls_back_to_index_not_outside(tmp_path, site, raw_path):
    (tmp_path / "outside").write_bytes(SECRET)
    exchange = await call(create_app(make_settings(tmp_path, frontend_dist=site)), [b""], path=raw_path, method="GET")
    assert exchange.status == 200 and exchange.body == INDEX


def test_dist_reached_through_a_symlink_still_serves_files(make_client, site, tmp_path):
    os.symlink(site, tmp_path / "dist-link")
    client = make_client(frontend_dist=tmp_path / "dist-link")
    assert client.get("/assets/app.js").content == SCRIPT


def test_without_a_build_everything_non_api_is_json_404(client):
    assert_error(client.get("/"), 404, "not_found")
    assert_error(client.get("/about"), 404, "not_found")
    assert client.get("/api/health").status_code == 200


def test_docs_are_not_exposed(spa_client):
    assert spa_client.get("/docs").content == INDEX
    assert_error(spa_client.get("/openapi.json"), 404, "not_found")
