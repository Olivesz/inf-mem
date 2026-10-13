# ==============================================================
# inf-mem — Makefile
# ==============================================================
# Personal LLM memory layer. Runs a local backend that the
# Chrome extension talks to for memory extraction and retrieval.
#
# QUICK START:
#   1. make install
#   2. Add your Groq key to config.yaml
#   3. make start
#   4. Load extension/  in chrome://extensions (developer mode)
#
# TARGETS:
#   make install    — create .venv and install dependencies into it
#   make test       — lint, unit and API tests (offline)
#   make test-model — tests that load the real embedding model
#   make calibrate  — recompute the gate calibration files in eval/results/
#   make eval-check — recompute them and compare with the committed files
#   make bench      — latency at 100, 1k and 10k memories
#   make start      — start the memory backend server
#   make dev        — start with auto-reload (development)
#   make stop       — kill the backend server
#   make status     — check if backend is running
#   make memories   — print all stored memories
#   make clear      — delete all stored memories (irreversible)
#   make clean      — remove pycache files
# ==============================================================

VENV        := .venv
PYTHON      := $(VENV)/bin/python
BACKEND_DIR := backend
PORT        := 7842
PID_FILE    := .backend.pid

.PHONY: install test test-model calibrate eval-check bench start dev stop status memories clear clean help

# ------------------------------------------------------------
# install: install all Python dependencies
# ------------------------------------------------------------
install:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install -q -r requirements-dev.txt

test:
	$(VENV)/bin/ruff check .
	$(PYTHON) -m pytest -q --cov=backend --cov-fail-under=85

test-model:
	$(PYTHON) -m pytest -q -m model tests/test_real_embedding.py

calibrate:
	$(PYTHON) eval/retrieval.py
	$(PYTHON) eval/dedupe.py

eval-check:
	$(PYTHON) eval/retrieval.py --check
	$(PYTHON) eval/dedupe.py --check

bench:
	$(PYTHON) bench/bench.py --out bench/results-local.json

# ------------------------------------------------------------
# start: launch the backend server in the background
# ------------------------------------------------------------
start:
	@echo "Starting inf-mem backend on port $(PORT)..."
	@cd $(BACKEND_DIR) && \
		../$(PYTHON) -m uvicorn main:app \
			--host 127.0.0.1 \
			--port $(PORT) \
			--log-level info \
		& echo $$! > ../$(PID_FILE)
	@sleep 1
	@$(MAKE) status

# ------------------------------------------------------------
# dev: start with auto-reload for development
# ------------------------------------------------------------
dev:
	@echo "Starting inf-mem backend in dev mode (auto-reload)..."
	@cd $(BACKEND_DIR) && \
		../$(PYTHON) -m uvicorn main:app \
			--host 127.0.0.1 \
			--port $(PORT) \
			--reload \
			--log-level debug

# ------------------------------------------------------------
# stop: kill the background server
# ------------------------------------------------------------
stop:
	@if [ -f $(PID_FILE) ]; then \
		kill $$(cat $(PID_FILE)) 2>/dev/null && echo "Backend stopped." || echo "Process not found."; \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found. Trying pkill..."; \
		pkill -f "uvicorn main:app" 2>/dev/null || echo "No process to kill."; \
	fi

# ------------------------------------------------------------
# status: check if backend is responding
# ------------------------------------------------------------
status:
	@curl -s http://127.0.0.1:$(PORT)/health | python3 -m json.tool 2>/dev/null \
		|| echo "Backend is not running. Use: make start"

# ------------------------------------------------------------
# memories: print all stored memories as JSON
# ------------------------------------------------------------
memories:
	@curl -s "http://127.0.0.1:$(PORT)/memories" | python3 -m json.tool \
		|| echo "Backend is not running. Use: make start"

# ------------------------------------------------------------
# clear: delete ALL stored memories (asks for confirmation)
# ------------------------------------------------------------
clear:
	@echo "WARNING: This will delete all stored memories from ~/.inf_mem/"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] \
		&& rm -f ~/.inf_mem/memories.db ~/.inf_mem/embeddings.npy ~/.inf_mem/memory_ids.npy \
		&& echo "All memories cleared." \
		|| echo "Cancelled."

# ------------------------------------------------------------
# clean: remove Python cache files
# ------------------------------------------------------------
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

# ------------------------------------------------------------
# help: show available targets
# ------------------------------------------------------------
help:
	@echo ""
	@echo "inf-mem — available targets"
	@echo "---------------------------"
	@echo "  make install    create .venv and install dependencies"
	@echo "  make test       lint, unit and API tests (offline)"
	@echo "  make test-model tests with the real embedding model"
	@echo "  make calibrate  recompute eval/results/ calibration files"
	@echo "  make eval-check recompute and compare with committed results"
	@echo "  make bench      latency at 100, 1k and 10k memories"
	@echo "  make start      start backend server (background)"
	@echo "  make dev        start with auto-reload (foreground)"
	@echo "  make stop       stop the background server"
	@echo "  make status     check backend health"
	@echo "  make memories   print all stored memories"
	@echo "  make clear      delete all stored memories"
	@echo "  make clean      remove __pycache__ files"
	@echo ""
