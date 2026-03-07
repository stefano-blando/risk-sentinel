PYTHON ?= ./venv/bin/python

.PHONY: test demo-check submission-bundle submission-audit azure-live-check

test:
	$(PYTHON) -m pytest -q

demo-check:
	$(PYTHON) scripts/demo_check.py --output artifacts/demo_check_latest.json

submission-bundle: demo-check
	$(PYTHON) scripts/build_submission_bundle.py --output-dir artifacts

submission-audit:
	$(PYTHON) scripts/submission_audit.py --output artifacts/submission_audit_latest.json

azure-live-check:
	RUN_AZURE_INTEGRATION_TESTS=1 $(PYTHON) -m pytest -q tests/test_azure_live_integration.py
