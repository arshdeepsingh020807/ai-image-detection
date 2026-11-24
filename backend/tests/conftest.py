from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from backend.app.main import create_app
from backend.tests.helpers import Reference, make_settings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def reference() -> Reference:
    return Reference()


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[Any]:
    """Factory for a started app (lifespan run) with test settings; every client is closed afterwards."""
    clients: list[TestClient] = []

    def build(**overrides: Any) -> TestClient:
        client = TestClient(create_app(make_settings(tmp_path, **overrides)))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Any) -> TestClient:
    return make_client()


@pytest.fixture(scope="module")
def shared_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Read-only tests share one started app to avoid reloading the model each time."""
    with TestClient(create_app(make_settings(tmp_path_factory.mktemp("shared")))) as shared:
        yield shared
