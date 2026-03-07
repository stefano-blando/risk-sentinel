PYTHON ?= ./venv/bin/python

.PHONY: test demo-check submission-bundle

test:
	$(PYTHON) -m pytest -q

demo-check:
	$(PYTHON) scripts/demo_check.py --output artifacts/demo_check_latest.json

submission-bundle: demo-check
	$(PYTHON) scripts/build_submission_bundle.py --output-dir artifacts
