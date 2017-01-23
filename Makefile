# Check that MPI path exists.
ifeq ("$(wildcard $(MPI_ROOT))","")
$(error Could not find MPI in "$(MPI_ROOT)")
endif

# Check that CUDA path exists.
ifeq ("$(wildcard $(CUDA_ROOT))","")
$(error Could not find CUDA in "$(CUDA_ROOT)")
endif

CC:=mpic++
NVCC:=nvcc
LDFLAGS:=-L$(CUDA_ROOT)/lib64 -L$(MPI_ROOT)/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX=
CFLAGS:=-std=c++11 -I$(MPI_ROOT)/include -I. -I$(CUDA_ROOT)/include -DOMPI_SKIP_MPICXX=
EXE_NAME:=allreduce-test
SRC:=$(wildcard *.cpp test/*.cpp)
CU_SRC:=$(wildcard *.cu)
OBJS:=$(SRC:.cpp=.o) $(CU_SRC:.cu=.o)

all: $(EXE_NAME)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@

$(EXE_NAME): $(OBJS)
	$(CC) -o $(EXE_NAME) $(LDFLAGS) $^ $(LDFLAGS)

test: $(EXE_NAME)
	$(EXE_NAME)

clean:
	rm -f *.o test/*.o $(EXE_NAME)
