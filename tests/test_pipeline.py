import pandas as pd

import classify
from processor_bert import classify_with_bert_batch
from processor_llm import parse_category
from processor_regex import classify_with_regex


def test_regex_matches_known_templates():
    assert classify_with_regex("User User123 logged in.") == "User Action"
    assert classify_with_regex("Backup completed successfully.") == "System Notification"
    assert classify_with_regex("Account with ID 1234 created by User1.") == "User Action"


def test_regex_returns_none_for_unknown_logs():
    assert classify_with_regex("Disk I/O latency above threshold on node 7") is None


def test_bert_tier_applies_confidence_threshold(fake_models):
    assert classify_with_bert_batch(["some log"]) == ["Critical Error"]
    fake_models(confidence=0.4)
    assert classify_with_bert_batch(["some log"]) == ["Unclassified"]


def test_bert_tier_handles_empty_batch():
    assert classify_with_bert_batch([]) == []


def test_parse_category_strips_reasoning_and_validates():
    assert parse_category("<category>Workflow Error</category>") == "Workflow Error"
    assert parse_category("<think>maybe <category>Deprecation Warning</category>?</think><category>Workflow Error</category>") == "Workflow Error"
    assert parse_category("<category>Something Else</category>") == "Unclassified"
    assert parse_category("no tags at all") == "Unclassified"
    assert parse_category(None) == "Unclassified"


def test_pipeline_routes_each_tier_and_keeps_order(fake_models, monkeypatch):
    llm_calls = []

    def fake_llm(message):
        llm_calls.append(message)
        return "Workflow Error"

    monkeypatch.setattr(classify, "classify_with_llm", fake_llm)

    logs = [
        ("ModernCRM", "Backup completed successfully."),        # regex
        ("LegacyCRM", "Lead conversion failed for ID 7842."),   # llm
        ("ModernHR", "Email service experiencing issues"),       # embeddings
        ("BillingSystem", "User User12 logged out."),            # regex
    ]
    labels = classify.classify(logs)

    assert labels == ["System Notification", "Workflow Error", "Critical Error", "User Action"]
    assert llm_calls == ["Lead conversion failed for ID 7842."]


def test_classify_dataframe_adds_label_column(fake_models):
    df = pd.DataFrame({"source": ["ModernCRM"], "log_message": ["Backup completed successfully."]})
    result = classify.classify_dataframe(df)
    assert list(result["target_label"]) == ["System Notification"]
    assert "target_label" not in df.columns  # input is not mutated
