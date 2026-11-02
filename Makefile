.PHONY: help install-local install-colab test test-cov lint format run-api run-ui clean

help:
	@echo "FraudSpotter Development Makefile"
	@echo "Commands:"
	@echo "  make install-local  - Install local development dependencies"
	@echo "  make install-colab  - Install Colab GPU training dependencies"
	@echo "  make test           - Run Pytest suite"
	@echo "  make test-cov       - Run tests with code coverage"
	@echo "  make lint           - Run linter (ruff)"
	@echo "  make format         - Format code using ruff"
	@echo "  make run-api        - Start FastAPI backend locally"
	@echo "  make run-ui         - Start Streamlit frontend locally"
	@echo "  make clean          - Clean temporary caches and build artifacts"

install-local:
	pip install -r requirements-local.txt

install-colab:
	pip install -r requirements-colab.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/fraudspotter --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

run-api:
	uvicorn src.fraudspotter.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run src/fraudspotter/web/app.py --server.port 8501

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

