.PHONY: install train score test lint app all

install:
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=src python -m churn.train

score:
	PYTHONPATH=src python -m churn.score

test:
	pytest

lint:
	ruff check src tests app

app:
	streamlit run app/streamlit_app.py

all: lint train score test
