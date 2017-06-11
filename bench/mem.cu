#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

/*
nvcc -std=c++11 --compiler-options -Wall -O2 \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_61,code=compute_61 \
	-o build/vmem mem.cu
*/

/* see also http://llvm.org/docs/CompileCudaWithLLVM.html
clang++ -std=c++11 -Wall -O2 \
	--cuda-gpu-arch=sm_30 \
	--cuda-gpu-arch=sm_35 \
	--cuda-gpu-arch=sm_50 \
	--cuda-gpu-arch=sm_52 \
	--cuda-gpu-arch=sm_60 \
	--cuda-gpu-arch=sm_61 \
	-o build/vmem mem.cu \
	-L/usr/local/cuda/lib64 \
	-lcudart_static -ldl -lrt -pthread
*/

// ref: https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
//      http://docs.nvidia.com/cuda/cuda-runtime-api/#axzz4jNvlr4KG
//      https://developer.nvidia.com/cuda-code-samples

static std::size_t N_THREADS;
static std::size_t MEM_LIMIT = 2ul << 30;

void init(int argc, char *argv[]){
	int deviceCount; cudaDeviceProp deviceProp;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess) std::exit(0);
	for (int device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&deviceProp, device);
		if (deviceProp.major == 9999) continue;
		if (device == 0) {
			printf("GPU processor count = %d\n", deviceProp.multiProcessorCount);
			printf("Threads per processor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
			N_THREADS = deviceProp.maxThreadsPerMultiProcessor;
		}
	}

	N_THREADS = 32;

	if(argc > 1){
		int tmp = atoi(argv[1]);
		if(tmp > 0) MEM_LIMIT = static_cast<std::size_t>(tmp) << 20;
	}

	MEM_LIMIT /= N_THREADS;
}

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
	  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) exit(code);
   }
}

__global__ void kernel_dummy(char *_buffer, std::size_t *_ret, std::size_t const MEM_LIMIT, std::size_t const BLK_SZ){
	curandState localState;
	curand_init(1234, threadIdx.x, 0, &localState);
	float *local_buffer = reinterpret_cast<float*>(_buffer+threadIdx.x*MEM_LIMIT);
	float const val = 2.71828f;
	std::size_t k = 0;
	for(;k<((MEM_LIMIT/BLK_SZ));++k){
		std::size_t random_block = static_cast<std::size_t>(curand_uniform(&localState) * ((MEM_LIMIT-BLK_SZ)/sizeof(float)));
		for(std::size_t p = 0;p<(MEM_LIMIT/sizeof(float));++p) // TODO BLK_SZ
			(local_buffer+random_block)[p] = val;
	}

	_ret[threadIdx.x] = k*BLK_SZ;
}

__global__ void kernel_sum_ret(std::size_t *_ret){
	extern __shared__ std::size_t sdata[];
	sdata[threadIdx.x] = _ret[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	for (std::size_t s = (blockDim.x>>1);s>0;s>>=1) {
		if(threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
		__syncthreads();
	}

	if (threadIdx.x == 0) _ret[blockIdx.x] = sdata[0];
}

void ramdom_write(std::size_t BLK_SZ){
	std::cout<<"Writing to "<<((MEM_LIMIT*N_THREADS)>>20)<<"MiB block with "<<
		N_THREADS<<" threads (BLK_SZ = "<<(BLK_SZ>>10)<<"KiB)..."<<std::endl;

	char *_buffer; cudaMalloc(&_buffer, MEM_LIMIT*N_THREADS);
	std::size_t *_ret; cudaMalloc(&_ret, sizeof(std::size_t)*N_THREADS);

	auto t1 = std::chrono::high_resolution_clock::now();
	kernel_dummy<<<1, N_THREADS>>>(_buffer, _ret, MEM_LIMIT, BLK_SZ);
	cudaDeviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();
	kernel_sum_ret<<<1, N_THREADS, N_THREADS*sizeof(std::size_t)>>>(_ret);
	std::size_t total_size = 0;
	cudaCheck(cudaMemcpy(&total_size, _ret, sizeof(std::size_t), cudaMemcpyDeviceToHost));
	std::cout<<"total_size="<<total_size<<std::endl;
	std::chrono::duration<double, std::milli> time_ra = t2 - t1;
	std::cout<<"  "<<time_ra.count()<<"ms @ "<<
		static_cast<float>(total_size>>30)/time_ra.count()*1000.f<<"GiB/s"<<std::endl;
	cudaFree(_buffer);
	cudaFree(_ret);
}

int main(int argc, char *argv[]){
	init(argc, argv);

	ramdom_write(8 << 20);
	ramdom_write(1 << 20);
	// ramdom_write(128 << 10);
	// ramdom_write(16 << 10);
	// ramdom_write(2 << 10);
	// ramdom_write(1 << 10);

	// ramdom_read(512 << 10);
	// ramdom_read(8 << 10);
	// ramdom_read(4 << 10);
	// ramdom_read(2 << 10);
	// ramdom_read(1 << 10);
	// ramdom_read(256);

	return 0;
}
