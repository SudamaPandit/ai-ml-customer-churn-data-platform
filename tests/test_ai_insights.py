from src.ai_insights import build_insight_prompt, summarize_with_bedrock


def test_prompt_contains_only_aggregate_metrics():
    prompt = build_insight_prompt({"row_count": 1000, "drift_score": 0.14})
    assert "row_count" in prompt
    assert "drift_score" in prompt
    assert "personal data" in prompt.lower()


def test_ai_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AI_ENABLED", raising=False)
    assert summarize_with_bedrock({"row_count": 10}) == "AI insights disabled"


def test_ai_enabled_without_model_id_fails_before_cloud_call(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    try:
        summarize_with_bedrock({"row_count": 10})
    except KeyError as exc:
        assert exc.args == ("BEDROCK_MODEL_ID",)
    else:
        raise AssertionError("Expected BEDROCK_MODEL_ID configuration error")
