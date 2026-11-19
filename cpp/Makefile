CXX      ?= clang++
CC       ?= cc
UNAME_M  := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
MCPU     ?= -mcpu=native
else ifeq ($(UNAME_M),aarch64)
MCPU     ?= -mcpu=native
else
MCPU     ?= -march=native
endif
OPT      ?= -O3 $(MCPU)
BUILD    ?= build
UNAME_S  := $(shell uname -s)
SHLIB    := $(BUILD)/libchaosys.$(if $(filter Darwin,$(UNAME_S)),dylib,so)
CXXSTD   := -std=c++20
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
COMMON   := $(CXXSTD) $(WARN) -DNDEBUG -Iinclude
LDLIBS   := -lpthread

HEADERS  := $(wildcard include/*.h include/*.hpp include/chaosys/*.hpp)
SRC_OBJS := $(BUILD)/bench.o $(BUILD)/baseline.o $(BUILD)/kernels.o \
            $(BUILD)/kernels_fast.o
LIB_OBJS := $(BUILD)/kernels.o
BIN      := $(BUILD)/chaosys_bench
CLI      := $(BUILD)/chaosys_cli
TEST_BIN := $(BUILD)/chaosys_tests

.PHONY: all run variants clean distclean quick test test-all test-asan cli smoke shared test-c test-exceptions test-time-header

all: $(BIN) $(CLI)

$(BUILD):
	@mkdir -p $(BUILD)

# The baseline deliberately gets an ordinary -O2 build: that is what "before"
# means here.  Everything else is compiled with $(OPT).
$(BUILD)/baseline.o: src/baseline.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) -O2 -c $< -o $@

$(BUILD)/kernels.o: src/kernels.cpp src/kernels_body.inc $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) $(OPT) -c $< -o $@

$(BUILD)/kernels_fast.o: src/kernels_fast.cpp src/kernels_body.inc $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) $(OPT) -ffast-math -c $< -o $@

$(BUILD)/bench.o: src/bench.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) $(OPT) -c $< -o $@

$(BIN): $(SRC_OBJS)
	$(CXX) $^ -o $@ $(LDLIBS)

$(CLI): apps/chaosys_cli.cpp $(LIB_OBJS) $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) $(OPT) $< $(LIB_OBJS) -o $@ $(LDLIBS)

cli: $(CLI)

$(TEST_BIN): tests/test_chaosys.cpp $(LIB_OBJS) $(HEADERS) \
             src/kernels_body.inc | $(BUILD)
	$(CXX) $(COMMON) $(OPT) $< $(LIB_OBJS) -o $@ $(LDLIBS)

shared: $(SHLIB)

$(SHLIB): src/c_api.cpp src/kernels.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(COMMON) $(OPT) -fPIC -shared src/c_api.cpp src/kernels.cpp \
	  -o $@ $(LDLIBS)

test-c: $(SHLIB)
	$(CC) -std=c11 -O2 -Wall -Wextra -Iinclude tests/test_c_api.c $(SHLIB) \
	  -o $(BUILD)/test_c_api -lm
	./$(BUILD)/test_c_api

test: $(TEST_BIN)
	./$(TEST_BIN)

# Compile a standalone test source against the shared library and run it.
define run_shared_test
	$(CXX) $(COMMON) -O2 tests/$(1).cpp $(SHLIB) -o $(BUILD)/$(1)
	./$(BUILD)/$(1)
endef

test-exceptions: $(SHLIB)
	$(call run_shared_test,test_exception_safety)

test-time-header: $(SHLIB)
	$(call run_shared_test,test_time_header)

test-all: test test-c test-exceptions test-time-header test-asan smoke

smoke: $(BIN) $(CLI)
	@./$(CLI) help >/dev/null
	@./$(CLI) systems >/dev/null
	@./$(CLI) traj --steps 100 --stride 10 >/dev/null
	@./$(CLI) section --steps 500 --max 3 >/dev/null
	@./$(CLI) lyapunov --transient 100 --segment 10 --segments 10 >/dev/null
	@./$(CLI) lyapunov --system rossler --transient 100 --segment 10 --segments 10 >/dev/null
	@./$(CLI) lyapunov --system chen --transient 100 --segment 10 --segments 10 >/dev/null
	@./$(CLI) dimension --steps 3000 --transient 500 --max-points 500 >/dev/null
	@./$(CLI) entropy --steps 3000 --transient 500 >/dev/null
	@./$(CLI) ensemble --n 512 --steps 8 --threads 2 >/dev/null
	@echo "CLI smoke OK"

# ASan + UBSan build of the same suite.  No -mcpu/-ffast-math so the sanitizer
# sees the plain scalar and NEON code paths.
test-asan: | $(BUILD)
	$(CXX) $(COMMON) -O1 -g -fno-omit-frame-pointer \
	  -fsanitize=address,undefined -fno-sanitize-recover=all \
	  tests/test_chaosys.cpp src/kernels.cpp \
	  -o $(BUILD)/chaosys_tests_asan $(LDLIBS)
	./$(BUILD)/chaosys_tests_asan

run: $(BIN)
	./$(BIN)

quick: $(BIN)
	./$(BIN) quick

# --- compiler-flag ablation -------------------------------------------------
# Each flag set is compiled into its own binary, which prints one CSV row.
variants: $(BIN)
	@mkdir -p results build/flags
	@printf 'variant,scalar_ns,neon1_mtps,neonN_mtps\n' > results/flags_raw.csv
	@for spec in "O2:-O2" "O3:-O3" "O3+native:-O3 $(MCPU)" \
	              "O3+native+fast-math:-O3 $(MCPU) -ffast-math"; do \
	  name=$${spec%%:*}; flags=$${spec#*:}; \
	  echo "  building $$name ($$flags)"; \
	  $(CXX) $(COMMON) $$flags src/bench.cpp src/baseline.cpp src/kernels.cpp \
	    src/kernels_fast.cpp -o build/flags/$$name $(LDLIBS); \
	  for i in 1 2 3; do ./build/flags/$$name flags $$name >> results/flags_raw.csv; done; \
	done
	@python3 scripts/merge_flags.py results/flags_raw.csv results/flags.csv

clean:
	rm -rf build

distclean: clean
	rm -rf results
