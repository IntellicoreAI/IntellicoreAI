from fastapi.testclient import TestClient
import app.main as main
from app.main import app
from app.drift import compare_windows
from app.compliance import fintech_checks

client = TestClient(app)

def test_feedback_is_saved_and_retrieved():
    event = client.post('/evaluate', json={'project_id':'feedback-test','prompt':'What is 2+2?','response':'4'}).json()
    response = client.post('/feedback', json={'event_id':event['event_id'],'vote':'up','issue_category':'helpful'})
    assert response.status_code == 200
    feedback = client.get('/feedback', params={'event_id':event['event_id']})
    assert feedback.status_code == 200
    assert feedback.json()['count'] >= 1

def test_drift_detected_for_shifted_window():
    result = compare_windows([10,11,9,10,10,11], [20,21,19,20,20,21], 2.0)
    assert result['drift_detected'] is True

def test_drift_requires_minimum_window():
    try:
        compare_windows([1,2,3], [4,5,6])
        assert False, 'Expected ValueError'
    except ValueError:
        pass

def test_drift_endpoint():
    response = client.post('/drift/check', json={'project_id':'p','metric_name':'latency','reference':[10,11,9,10,10],'current':[30,31,29,30,30]})
    assert response.status_code == 200 and response.json()['drift_detected']

def test_fintech_checks_flag_sensitive_identifier():
    result = fintech_checks('Please review this', 'PAN ABCDE1234F is on file')
    assert 'possible_indian_pan' in result['flags']

def test_alerts_endpoint():
    response = client.get('/alerts')
    assert response.status_code == 200
    assert 'alerts' in response.json()

def test_provider_chat_telemetry(monkeypatch):
    def fake_chat(*args, **kwargs):
        return {'response':'Your balance is 100.','model_id':'fake-model','latency_ms':120.0,
                'input_tokens':8,'output_tokens':6,'total_tokens':14,'provider_response_id':'fake-id','raw_usage':{}}
    monkeypatch.setattr(main, 'chat_completion', fake_chat)
    response = client.post('/chat/completions', json={'project_id':'provider-test','prompt':'What is my balance?'})
    assert response.status_code == 200
    body=response.json()
    assert body['response'] == 'Your balance is 100.'
    assert body['usage']['total_tokens'] == 14
    assert body['monitoring']['event_id']

def test_semantic_endpoint_uses_embeddings(monkeypatch):
    monkeypatch.setattr(main, 'create_embeddings', lambda texts, model=None: [[1.0,0.0],[0.9,0.1]])
    response=client.post('/evaluate/semantic',json={'reference_text':'car','candidate_text':'automobile'})
    assert response.status_code == 200
    assert response.json()['semantic_similarity'] > 0.99

def test_saved_event_matches_shared_base_event_contract():
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator, FormatChecker
    schema_path = Path(__file__).resolve().parents[4] / 'shared' / 'schemas' / 'events' / 'base-event.schema.json'
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    event = client.post('/evaluate', json={'project_id':'schema-test','prompt':'Explain EMI','response':'EMI is a periodic loan payment.'}).json()
    stored = client.get('/events', params={'project_id':'schema-test'}).json()['events'][0]
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(stored)
    assert stored['system_type'] == 'LLM'
    assert stored['event_id'] == event['event_id']
