from agent.tools.web import web_search_schema


def test_schema_shape():
    assert web_search_schema() == {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


def test_max_uses():
    assert web_search_schema(max_uses=2)["max_uses"] == 2
