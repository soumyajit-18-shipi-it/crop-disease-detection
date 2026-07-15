from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.db.seed_disease_data import seed_database
from backend.main import app


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Keep backend tests away from the repository database."""
    database_path = tmp_path / "test_disease_info.db"
    monkeypatch.setattr(settings, "db_path", database_path)
    seed_database(database_path)
    return database_path


@pytest.fixture
def client(isolated_database):
    """Create the API client only after the temporary DB path is active."""
    return TestClient(app)
