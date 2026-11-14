from llm_gateway.config import PROJECT_ROOT, Settings


def test_health_without_api_key(make_client):
    response = make_client().get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama_url"] == "http://127.0.0.1:11434"
    assert body["judge_model"] == "claude-opus-5-5"
    assert body["evaluator_configured"] is False
    assert body["environment"] in {"wsl", "linux", "other"}


def test_health_never_exposes_api_key(make_client):
    secret = "sk-ant-test-not-a-real-key"
    response = make_client(anthropic_api_key=secret).get("/api/health")

    assert response.json()["evaluator_configured"] is True
    assert secret not in response.text


def test_blank_api_key_means_not_configured():
    assert Settings(_env_file=None, anthropic_api_key="  ").evaluator_configured is False


def test_api_key_hidden_in_repr():
    settings = Settings(_env_file=None, anthropic_api_key="sk-ant-test-not-a-real-key")
    assert "sk-ant-test" not in repr(settings)


def test_relative_database_path_resolves_from_project_root():
    settings = Settings(_env_file=None, database_path="data/x.db")
    assert settings.database_path == PROJECT_ROOT / "data" / "x.db"
