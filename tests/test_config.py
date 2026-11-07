from agent.config import Config


def test_rag_defaults(tmp_path):
    config = Config(data_dir=tmp_path / "d", workspace=tmp_path / "w", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    assert config.rag_index_dir == tmp_path / "d" / "index"
    assert config.rag_max_chunk_tokens == 512
    assert (config.rag_candidates, config.rag_rrf_k, config.rag_top_k) == (20, 60, 8)
    assert config.rag_rerank is False


def test_rag_settings_read_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_RAG_RERANK", "true")
    monkeypatch.setenv("AGENT_RAG_TOP_K", "3")
    config = Config(data_dir=tmp_path / "d", workspace=tmp_path / "w", log_dir=tmp_path / "l", knowledge_dir=tmp_path / "k")
    assert config.rag_rerank is True and config.rag_top_k == 3
