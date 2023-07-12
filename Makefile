include .env

lint:
	@poetry run mypy language_model
	@poetry run pylint language_model
	@poetry run flake8 language_model
	@poetry run black language_model --check


test:
	@poetry run pytest


tox:
	@poetry run tox


requirements:
	@poetry export -f requirements.txt --output requirements.txt
	@poetry export -f requirements.txt --output requirements.dev.txt --dev


clean:
	@rm -rf .mypy_cache
	@rm -rf .tox
	@rm -rf .pytest_cache
