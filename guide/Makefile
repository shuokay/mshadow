# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include config.mk
include ../make/mshadow.mk
export CFLAGS = -Wall -O3 -std=c++11 -I../ $(MSHADOW_CFLAGS) $(ADD_MSHADOW_CFLAGS)
export LDFLAGS= -lm $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

# specify tensor path
BIN = basic defop test
OBJ =
CUOBJ =
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

basic: basic.cpp
defop: defop.cpp
basic_stream: basic_stream.cu
test: test.cpp
$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
