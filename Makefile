# =============================================================================
# Chimera Agent Baseline
# =============================================================================
#
# Local:
#   make install                    - editable install with dev deps
#   make run                        - run the agent locally
#   make test                       - run tests
#   make lint                       - check formatting + linting
#
# Grand Challenge:
#   make gc-build                   - build GC Docker image
#   make gc-test                    - local test run (mirrors GC runtime)
#   make gc-save                    - export image + model tarballs
#
# Cluster:
#   make submit                     - sync + submit job
#   make dev                        - sync + launch dev session
#   make status / logs / cancel     - monitor SLURM jobs
# =============================================================================

-include .env

# --- Defaults (override in .env or on CLI) ---
PROJECT_SLUG    ?= chimera_agent_baseline
REMOTE_HOST     ?= oaks-lab
DOCKER_IMAGE    ?= dockerdex.umcn.nl:5005/sandermoon/chimera_agent_baseline:latest

# --- NAS paths ---
NAS_BASE        ?= /data/pathology/projects/sander/projects
NAS_PATH        ?= $(NAS_BASE)/$(PROJECT_SLUG)
NAS_CODE        ?= $(NAS_PATH)/code
NAS_OUTPUTS     ?= $(NAS_PATH)/outputs
NAS_LOGS        ?= $(NAS_PATH)/logs
CONTAINER_MOUNTS ?= /data/pathology/projects:/data/pathology/projects,/data/pathology/archives:/data/pathology/archives,/data/pa_cpgarchive:/data/pa_cpgarchive

# Local NAS mount (macOS)
NAS_MOUNT       ?= /Volumes/temporary/sander/projects
LOCAL_NAS_PATH  ?= $(NAS_MOUNT)/$(PROJECT_SLUG)

# --- Tier config ---
TIER     ?= medium
QOS      ?= high
NODELIST ?=
GRES     ?=
CPUS     ?=
MEM      ?=
TIME     ?=

_tier_config := $(wildcard configs/nodes/$(TIER).sh)
ifneq ($(_tier_config),)
  include configs/nodes/$(TIER).sh
endif

RUN_ARGS   ?=
SLURM_MAIL ?=
SBATCH_MAIL_ARGS := $(if $(SLURM_MAIL),--mail-user=$(SLURM_MAIL),)

GC_IMAGE_TAG ?= chimera-agent-baseline

.PHONY: help install process-guidelines run test lint format lock sync submit dev logs logs-all status cancel gpu-status gc-build gc-test gc-test-cpu gc-save clean doctor init-nas

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Local development
# =============================================================================

install: ## Install project in editable mode with dev deps
	uv pip install -e ".[dev]"

process-guidelines: ## Process guidelines PDF into ChromaDB + save embedding model
	python scripts/process_guidelines.py

run: ## Run the agent locally
	python -m chimera_agent_baseline.run $(RUN_ARGS)

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

lock: ## Pin dependencies to requirements.lock
	uv pip freeze --exclude-editable > requirements.lock
	@echo "Pinned $$(wc -l < requirements.lock | tr -d ' ') packages to requirements.lock"

# =============================================================================
# Code sync
# =============================================================================

sync: ## Rsync code to NAS
	@if [ -d "$(NAS_MOUNT)" ]; then \
		echo "Syncing to $(LOCAL_NAS_PATH)/code/ (local mount)..."; \
		rsync -avz --delete --exclude-from=.syncignore . "$(LOCAL_NAS_PATH)/code/"; \
	else \
		echo "Syncing to $(NAS_CODE) (via SSH)..."; \
		rsync -avz --delete --exclude-from=.syncignore -e ssh . "$(REMOTE_HOST):$(NAS_CODE)/"; \
	fi
	@echo "Done."

# =============================================================================
# Cluster: job submission
# =============================================================================

submit: sync ## Sync + submit job (TIER=high GRES=gpu:1 ...)
	ssh $(REMOTE_HOST) "cd $(NAS_CODE) && sbatch \
		--job-name=$(PROJECT_SLUG) \
		--output=$(NAS_LOGS)/slurm-%j.out \
		--error=$(NAS_LOGS)/slurm-%j.err \
		--nodes=1 \
		--nodelist=$(NODELIST) \
		--gres=$(GRES) \
		--cpus-per-task=$(CPUS) \
		--mem=$(MEM) \
		--time=$(TIME) \
		--qos=$(QOS) \
		$(SBATCH_MAIL_ARGS) \
		--container-image=$(DOCKER_IMAGE) \
		--container-mounts=$(CONTAINER_MOUNTS) \
		--container-workdir=$(NAS_CODE) \
		--export=ALL,NAS_PATH=$(NAS_PATH),NAS_OUTPUTS=$(NAS_OUTPUTS),RUN_ARGS='$(RUN_ARGS)' \
		scripts/slurm/run.sbatch"

# =============================================================================
# Cluster: dev sessions
# =============================================================================

dev: sync ## Sync + launch interactive dev session (TIER=high ...)
	ssh $(REMOTE_HOST) "cd $(NAS_CODE) && sbatch \
		--job-name=dev-$(PROJECT_SLUG) \
		--output=$(NAS_LOGS)/slurm-%j.out \
		--error=$(NAS_LOGS)/slurm-%j.err \
		--nodes=1 \
		--nodelist=$(NODELIST) \
		--gres=$(GRES) \
		--cpus-per-task=$(CPUS) \
		--mem=$(MEM) \
		--time=12:00:00 \
		--qos=$(QOS) \
		--container-image=$(DOCKER_IMAGE) \
		--container-mounts=$(CONTAINER_MOUNTS) \
		--container-workdir=$(NAS_CODE) \
		--export=ALL,NAS_PATH=$(NAS_PATH),NAS_OUTPUTS=$(NAS_OUTPUTS) \
		scripts/slurm/dev.sbatch"

# =============================================================================
# Monitoring
# =============================================================================

logs: ## Tail SLURM job output (JOB=id for specific job, default: latest)
	@if [ -d "$(NAS_MOUNT)" ]; then \
		JOB=$${JOB:-$$(ls "$(LOCAL_NAS_PATH)/logs/slurm-"*.out 2>/dev/null | grep -o '[0-9]*' | sort -n | tail -1)}; \
		echo "=== Job $${JOB} ===" && \
		tail -f "$(LOCAL_NAS_PATH)/logs/slurm-$${JOB}.out" "$(LOCAL_NAS_PATH)/logs/slurm-$${JOB}.err"; \
	else \
		ssh $(REMOTE_HOST) "JOB=$${JOB:-\$$(ls $(NAS_LOGS)/slurm-*.out 2>/dev/null | grep -o '[0-9]*' | sort -n | tail -1)} && echo '=== Job \$${JOB} ===' && tail -f $(NAS_LOGS)/slurm-\$${JOB}.out $(NAS_LOGS)/slurm-\$${JOB}.err"; \
	fi

logs-all: ## Tail all running SLURM job outputs
	@JOBS=$$(ssh $(REMOTE_HOST) "squeue -u \$$USER -h -o '%i'" 2>/dev/null); \
	if [ -z "$$JOBS" ]; then echo "No running jobs."; exit 0; fi; \
	FILES=""; \
	for JOB in $$JOBS; do \
		FILES="$$FILES $(NAS_LOGS)/slurm-$${JOB}.out $(NAS_LOGS)/slurm-$${JOB}.err"; \
	done; \
	if [ -d "$(NAS_MOUNT)" ]; then \
		tail -f $$(echo "$$FILES" | sed "s|$(NAS_LOGS)|$(LOCAL_NAS_PATH)/logs|g"); \
	else \
		ssh $(REMOTE_HOST) "tail -f $$FILES"; \
	fi

status: ## Show your SLURM jobs
	@ssh $(REMOTE_HOST) "squeue -u \$$USER -o '%.10i %.20j %.8T %.10M %.6D %.4C %.10m %R'"

cancel: ## Cancel most recent job (with confirmation)
	@JOB=$$(ssh $(REMOTE_HOST) "squeue -u \$$USER -h -o '%i %j %T' | head -1") && \
	echo "  Cancel job: $$JOB" && \
	read -p "  Proceed? [y/N] " confirm && \
	[ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ] || { echo "Aborted."; exit 1; } && \
	JOB_ID=$$(echo "$$JOB" | awk '{print $$1}') && \
	ssh $(REMOTE_HOST) "scancel $$JOB_ID" && \
	echo "  Cancelled."

gpu-status: ## Check GPU availability across cluster
	@ssh $(REMOTE_HOST) "sinfo -N -l --partition=gpu"

# =============================================================================
# Environment check
# =============================================================================

doctor: ## Check that all prerequisites are working
	@echo "Checking environment..."
	@echo ""
	@printf "  %-30s" "Python" && python3 --version 2>/dev/null && echo "" || echo "not found"
	@printf "  %-30s" "uv" && uv --version 2>/dev/null || echo "not found"
	@printf "  %-30s" "Project importable" && python3 -c "import $(PROJECT_SLUG)" 2>/dev/null && echo "ok" || echo "run make install"
	@printf "  %-30s" "SSH to cluster" && ssh -o ConnectTimeout=5 $(REMOTE_HOST) "echo ok" 2>/dev/null || echo "check SSH config for $(REMOTE_HOST)"
	@printf "  %-30s" "NAS mounted" && ([ -d "$(NAS_MOUNT)" ] && echo "$(NAS_MOUNT)") || echo "NAS not mounted at $(NAS_MOUNT)"
	@printf "  %-30s" "NAS project dir" && ([ -d "$(LOCAL_NAS_PATH)" ] && echo "ok") || echo "run make init-nas"
	@echo ""

# =============================================================================
# NAS
# =============================================================================

init-nas: ## Create project folder structure on NAS
	@if [ -d "$(NAS_MOUNT)" ]; then \
		mkdir -p "$(LOCAL_NAS_PATH)/outputs" \
		         "$(LOCAL_NAS_PATH)/logs" \
		         "$(LOCAL_NAS_PATH)/code"; \
		echo "Created $(LOCAL_NAS_PATH)/{outputs,logs,code}"; \
	else \
		echo "NAS not mounted at $(NAS_MOUNT)"; \
		echo "Creating via SSH..."; \
		ssh $(REMOTE_HOST) "mkdir -p \
			$(NAS_PATH)/outputs \
			$(NAS_PATH)/logs \
			$(NAS_PATH)/code"; \
		echo "Created $(NAS_PATH)/{outputs,logs,code}"; \
	fi

# =============================================================================
# Grand Challenge
# =============================================================================

gc-build: ## Build Grand Challenge Docker image
	docker build --platform=linux/amd64 \
		--tag $(GC_IMAGE_TAG) .

GC_INPUT_DIR ?= test/input/task1

gc-test: gc-build ## Run local test mirroring GC runtime (no network, non-root)
	@mkdir -p test/output && chmod 777 test/output
	@rm -rf test/output/*
	docker run --rm \
		--network none \
		--gpus all \
		--volume $(CURDIR)/$(GC_INPUT_DIR):/input:ro \
		--volume $(CURDIR)/test/output:/output \
		--volume $(CURDIR)/model:/opt/ml/model:ro \
		$(GC_IMAGE_TAG)
	@echo ""
	@echo "Output files:"
	@ls -la test/output/

gc-test-cpu: gc-build ## Run local test without GPU
	@mkdir -p test/output && chmod 777 test/output
	@rm -rf test/output/*
	docker run --rm \
		--network none \
		--volume $(CURDIR)/$(GC_INPUT_DIR):/input:ro \
		--volume $(CURDIR)/test/output:/output \
		--volume $(CURDIR)/model:/opt/ml/model:ro \
		$(GC_IMAGE_TAG)
	@echo ""
	@echo "Output files:"
	@ls -la test/output/

gc-save: gc-build ## Export GC image + model as tarballs for upload
	docker save $(GC_IMAGE_TAG) | gzip -c > $(GC_IMAGE_TAG).tar.gz
	@echo "Saved $(GC_IMAGE_TAG).tar.gz ($$(du -h $(GC_IMAGE_TAG).tar.gz | cut -f1))"
	@if [ -n "$$(ls -A model/ 2>/dev/null)" ]; then \
		tar -czf model.tar.gz -C model . && \
		echo "Saved model.tar.gz ($$(du -h model.tar.gz | cut -f1))"; \
	else \
		echo "model/ is empty — skipping model.tar.gz"; \
	fi


# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove local build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf outputs/ logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned local artifacts."
