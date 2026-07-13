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
#   make install    — install Python dependencies
#   make start      — start the memory backend server
#   make dev        — start with auto-reload (development)
#   make stop       — kill the backend server
#   make status     — check if backend is running
#   make memories   — print all stored memories
#   make clear      — delete all stored memories (irreversible)
#   make clean      — remove pycache files
# ==============================================================

PYTHON      := python3
PIP         := pip3
BACKEND_DIR := backend
PORT        := 7842
PID_FILE    := .backend.pid

.PHONY: install start dev stop status memories clear clean help

# ------------------------------------------------------------
# install: install all Python dependencies
# ------------------------------------------------------------
install:
	$(PIP) install -r requirements.txt --break-system-packages

# ------------------------------------------------------------
# start: launch the backend server in the background
# ------------------------------------------------------------
start:
	@echo "Starting inf-mem backend on port $(PORT)..."
	@cd $(BACKEND_DIR) && \
		$(PYTHON) -m uvicorn main:app \
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
		$(PYTHON) -m uvicorn main:app \
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
	@curl -s http://127.0.0.1:$(PORT)/health | $(PYTHON) -m json.tool 2>/dev/null \
		|| echo "Backend is not running. Use: make start"

# ------------------------------------------------------------
# memories: print all stored memories as JSON
# ------------------------------------------------------------
memories:
	@curl -s "http://127.0.0.1:$(PORT)/memories" | $(PYTHON) -m json.tool \
		|| echo "Backend is not running. Use: make start"

# ------------------------------------------------------------
# clear: delete ALL stored memories (asks for confirmation)
# ------------------------------------------------------------
clear:
	@echo "WARNING: This will delete all stored memories from ~/.inf_mem/"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] \
		&& rm -f ~/.inf_mem/memories.db ~/.inf_mem/embeddings.npy \
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
	@echo "  make install    install Python dependencies"
	@echo "  make start      start backend server (background)"
	@echo "  make dev        start with auto-reload (foreground)"
	@echo "  make stop       stop the background server"
	@echo "  make status     check backend health"
	@echo "  make memories   print all stored memories"
	@echo "  make clear      delete all stored memories"
	@echo "  make clean      remove __pycache__ files"
	@echo ""
