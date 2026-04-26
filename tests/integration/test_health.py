from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import main


def test_health_endpoint(monkeypatch, client):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    monkeypatch.setattr(
        main,
        "SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    monkeypatch.setattr(main, "load_model", lambda: object())

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "db_connected": True}
