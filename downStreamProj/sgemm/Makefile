################################################################# For Debug  #################################################################

BINARY_NAME = app
CUDA_PATH   = /usr/local/cuda-12.4
CC          = $(CUDA_PATH)/bin/nvcc

# 调试编译选项
CFLAGS      = -O0 -g -G -Xcompiler -rdynamic -DUSE_MULTI_GPU
PTXASFLAGS  = -Xptxas -O0,-v
FASTMATH    = 
ARCHFLAGS   = -gencode arch=compute_86,code=sm_86

INCFLAGS    = -I$(CUDA_PATH)/include -I../../common/csrc/include -Icsrc/include -Icuda-samples/Common
LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand

SRC = $(wildcard csrc/*.cu)
OBJ = $(SRC:csrc/%.cu=build/%.o)

all: outputs/$(BINARY_NAME)

outputs/$(BINARY_NAME): $(OBJ)
	@mkdir -p outputs
	$(CC) $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(LDFLAGS) $^ -o $@

build/%.o: csrc/%.cu
	@mkdir -p build
	$(CC) -c $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(INCFLAGS) $< -o $@

clean:
	rm -rf build outputs



################################################################# For optimal performance  #################################################################

# BINARY_NAME = app
# CUDA_PATH   = /usr/local/cuda-12.4
# CC          = $(CUDA_PATH)/bin/nvcc

# # 编译选项
# CFLAGS      = -O3 -Xcompiler -fno-strict-aliasing -DUSE_MULTI_GPU
# PTXASFLAGS  = -Xptxas -O3,-v
# FASTMATH    = -use_fast_math
# ARCHFLAGS   = -gencode arch=compute_80,code=sm_80

# INCFLAGS    = -I$(CUDA_PATH)/include -I../../common/csrc/include -Icsrc/include -Icuda-samples/Common
# LDFLAGS     = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lcurand

# SRC = $(wildcard csrc/*.cu)
# OBJ = $(SRC:csrc/%.cu=build/%.o)

# all: outputs/$(BINARY_NAME)

# outputs/$(BINARY_NAME): $(OBJ)
# 	@mkdir -p outputs
# 	$(CC) $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(LDFLAGS) $^ -o $@

# build/%.o: csrc/%.cu
# 	@mkdir -p build
# 	$(CC) -c $(CFLAGS) $(PTXASFLAGS) $(FASTMATH) $(ARCHFLAGS) $(INCFLAGS) $< -o $@

# clean:
# 	rm -rf build outputs

