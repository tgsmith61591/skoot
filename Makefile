# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows

PYTHON ?= python

.PHONY: clean develop test install bdist_wheel

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf *.egg-info

deploy-requirements:
	$(PYTHON) -m pip install twine readme_renderer[md]

# Depends on an artifact existing in dist/, and two environment variables
deploy-twine-test: bdist_wheel deploy-requirements
	$(PYTHON) -m twine upload \
		--repository-url https://test.pypi.org/legacy/ dist/* \
		--username ${TWINE_USERNAME} \
		--password ${TWINE_PASSWORD}

doc-requirements:
	$(PYTHON) -m pip install -r build_tools/doc/doc_requirements.txt

documentation: doc-requirements
	@make -C doc clean html EXAMPLES_PATTERN=example_*

requirements:
	$(PYTHON) -m pip install -r requirements.txt

bdist_wheel: requirements
	$(PYTHON) setup.py bdist_wheel

sdist: requirements
	$(PYTHON) setup.py sdist

develop: requirements
	$(PYTHON) setup.py develop

install: requirements
	$(PYTHON) setup.py install

test-requirements:
	$(PYTHON) -m pip install pytest flake8

coverage-dependencies:
	$(PYTHON) -m pip install coverage pytest-cov codecov

test-lint: test-requirements
	$(PYTHON) -m flake8 skoot \
		--filename='*.py' \
		--exclude='skoot/__config__.py,skoot/_build_utils/system_info.py' \
		--ignore E803,F401,F403,W293,W504,W605

test-unit: test-requirements coverage-dependencies
	$(PYTHON) -m pytest -v --durations=20 --cov-config .coveragerc --cov skoot -p no:logging

test: develop test-unit test-lint
	# Coverage creates all these random little artifacts we don't want
	rm .coverage.* || echo "No coverage artifacts to remove"

twine-check: bdist_wheel deploy-requirements
	# Check that twine will parse the README acceptably
	$(PYTHON) -m twine check dist/*