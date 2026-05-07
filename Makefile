SHELL := /usr/bin/env bash

UV ?= uv
PYTHON ?= 3.12

# CI (and local stability) expects torch 2.11.0 for this repo.
TORCH_VERSION ?= 2.11.0

CONDA ?= conda
CONDA_BASE := $(shell $(CONDA) info --base 2>/dev/null)
CAGRA_CONDA_ENV ?= fastplaid-cagra
CAGRA_PREFIX := $(CONDA_BASE)/envs/$(CAGRA_CONDA_ENV)

TORCH_LIB := .venv/lib/python3.12/site-packages/torch/lib

CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDE ?= $(CUDA_HOME)/targets/sbsa-linux/include

LD_CAGRA := $(CAGRA_PREFIX)/lib:$(CAGRA_PREFIX)/targets/sbsa-linux/lib:$(TORCH_LIB)
LIB_CAGRA := $(CAGRA_PREFIX)/lib:$(TORCH_LIB):$(CUDA_HOME)/targets/sbsa-linux/lib:$(CUDA_HOME)/lib64
LIB_BASE := $(TORCH_LIB)

.PHONY: env env-dev install test test-all test-cagra conda-cagra conda-cagra-update build-base build-cagra wheel-cagra cagra-probe lint evaluate evaluate-test clean

env:
	$(UV) venv --allow-existing --managed-python --python $(PYTHON) .venv
	$(UV) pip install "torch==$(TORCH_VERSION)" maturin setuptools numpy cmake
	$(UV) pip install pytest pytest-cov

# Build a baseline extension (no cuVS dependency).
build-base: env
	env \
	  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER \
	  LIBRARY_PATH="$(LIB_BASE):$${LIBRARY_PATH:-}" \
	  LD_LIBRARY_PATH="$(TORCH_LIB):$${LD_LIBRARY_PATH:-}" \
	  CMAKE="$(PWD)/.venv/bin/cmake" \
	  $(UV) run --no-sync maturin develop --release

env-dev: env
	# Dev extras may require system headers; keep separate.
	$(UV) pip install ".[dev]" --no-build-isolation

install: build-base

test:
	$(MAKE) build-base
	$(UV) run --no-sync pytest tests/test.py -vv

test-all:
	$(MAKE) test
	$(MAKE) test-cagra

conda-cagra:
	@if [ -z "$(CONDA_BASE)" ]; then echo "conda not found (CONDA_BASE empty)"; exit 1; fi
	@if [ -d "$(CAGRA_PREFIX)" ]; then \
	  echo "conda env exists: $(CAGRA_PREFIX)"; \
	else \
	  $(CONDA) create -y -n $(CAGRA_CONDA_ENV) -c rapidsai -c nvidia -c conda-forge \
	    "libcuvs=26.04.*" "libraft=26.04.*" "librmm=26.04.*" rapids-logger cuda-cccl_linux-aarch64; \
	fi

conda-cagra-update:
	@if [ -z "$(CONDA_BASE)" ]; then echo "conda not found (CONDA_BASE empty)"; exit 1; fi
	@if [ ! -d "$(CAGRA_PREFIX)" ]; then \
	  echo "conda env missing: $(CAGRA_PREFIX)"; \
	  echo "Run: make conda-cagra"; \
	  exit 1; \
	fi
	$(CONDA) install -y -n $(CAGRA_CONDA_ENV) -c rapidsai -c nvidia -c conda-forge \
	  "libcuvs=26.04.*" "libraft=26.04.*" "librmm=26.04.*" rapids-logger cuda-cccl_linux-aarch64

build-cagra: env conda-cagra
	# Build the Rust extension with cuVS/CAGRA enabled.
	env \
	  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER \
	  CUDA_HOME="$(CUDA_HOME)" \
	  BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/aarch64-linux-gnu/13/include -I/usr/lib/gcc/aarch64-linux-gnu/13/include-fixed -I/usr/include/aarch64-linux-gnu -I$(CUDA_INCLUDE) -I$(CUDA_HOME)/include" \
	  CMAKE_PREFIX_PATH="$(CAGRA_PREFIX)" \
	  cuvs_DIR="$(CAGRA_PREFIX)/lib/cmake/cuvs" \
	  LIBRARY_PATH="$(LIB_CAGRA):$${LIBRARY_PATH:-}" \
	  LD_LIBRARY_PATH="$(LD_CAGRA):$${LD_LIBRARY_PATH:-}" \
	  CMAKE="$(PWD)/.venv/bin/cmake" \
	  $(UV) run --no-sync maturin develop --release --features cagra

# Build a distributable wheel with cuVS/CAGRA enabled into target/wheels/.
wheel-cagra: env conda-cagra
	@mkdir -p target/wheels
	@rm -f target/wheels/fast_plaid-*.whl
	env \
	  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER \
	  CUDA_HOME="$(CUDA_HOME)" \
	  BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/aarch64-linux-gnu/13/include -I/usr/lib/gcc/aarch64-linux-gnu/13/include-fixed -I/usr/include/aarch64-linux-gnu -I$(CUDA_INCLUDE) -I$(CUDA_HOME)/include" \
	  CMAKE_PREFIX_PATH="$(CAGRA_PREFIX)" \
	  cuvs_DIR="$(CAGRA_PREFIX)/lib/cmake/cuvs" \
	  LIBRARY_PATH="$(LIB_CAGRA):$${LIBRARY_PATH:-}" \
	  LD_LIBRARY_PATH="$(LD_CAGRA):$${LD_LIBRARY_PATH:-}" \
	  CMAKE="$(PWD)/.venv/bin/cmake" \
	  $(UV) run --no-sync maturin build --release --features cagra --auditwheel skip -o target/wheels

test-cagra: build-cagra
	env \
	  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER \
	  LD_LIBRARY_PATH="$(LD_CAGRA):$${LD_LIBRARY_PATH:-}" \
	  $(UV) run --no-sync pytest tests/test.py -m cagra -vv

# Synthetic index + search; bisects centroid CAGRA probe behavior (padding, batch size, scale).
cagra-probe: build-cagra
	env \
	  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER \
	  LD_LIBRARY_PATH="$(LD_CAGRA):$${LD_LIBRARY_PATH:-}" \
	  $(UV) run --no-sync python scripts/cagra_centroid_probe_repro.py --sweep

lint: env-dev
	$(UV) run --no-sync pre-commit run --files python/**/**/**.py

evaluate:
	$(UV) run --no-sync python docs/benchmark/benchmark.py
	rm -rf *.dat
	
evaluate-test:
	mprof run --include-children $(UV) run --no-sync python test.py && mprof plot -o chart_test.png
	rm -rf *.dat	

clean:
	cargo clean
	rm -rf .venv .pytest_cache
