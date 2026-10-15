.PHONY: install test ingest eval ask

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	EMBEDDING_BACKEND=hash python -m pytest -q

ingest:
	python -m rag_agent ingest

eval: ingest
	python -m rag_agent eval

ask:
	python -m rag_agent ask "$(Q)" --trace
