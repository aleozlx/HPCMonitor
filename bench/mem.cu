// #include <boost/thread.hpp>
// #include <boost/shared_ptr.hpp>
// #include <boost/make_shared.hpp>
#include <iostream>
#include <chrono>
#include <vector>
// #include <cstdlib>
// #include <ctime>

// nvcc -std=c++11 --compiler-options -Wall -O2 -gencode arch=compute_60,code=compute_60 -o build/vmem mem.cu
// see also http://llvm.org/docs/CompileCudaWithLLVM.html

// typedef struct {
// 	boost::shared_ptr<boost::thread> handle;
// 	char *local_buffer;
// 	std::size_t ret;
// } thread_rc;

void init(int argc, char *argv[]){
// 	N_THREADS = boost::thread::hardware_concurrency();

// 	if(argc > 1){
// 		int tmp = atoi(argv[1]);
// 		if(tmp > 0) MEM_LIMIT = static_cast<std::size_t>(tmp) << 20;
// 	}

// 	std::cout<<"Page size "<<getpagesize()<<"B"<<std::endl;

// 	MEM_LIMIT /= N_THREADS;
// 	std::srand(std::time(0));
}

void ramdom_write(std::size_t BLK_SZ){
// 	std::cout<<"Writing to "<<((MEM_LIMIT*N_THREADS)>>20)<<"MiB block with "<<
// 		N_THREADS<<" threads (BLK_SZ = "<<(BLK_SZ>>10)<<"KiB)..."<<std::endl;

// 	thread_rc threads[N_THREADS];
// 	for(std::size_t i=0;i<N_THREADS;++i)
// 		threads[i].local_buffer = new char[MEM_LIMIT];

// 	auto t1 = std::chrono::high_resolution_clock::now();
// 	for(std::size_t i=0;i<N_THREADS;++i){
// 		threads[i].handle = boost::make_shared<boost::thread>([&,i](){
// 			std::size_t k = 0;
// 			for(;k<((MEM_LIMIT/BLK_SZ)<<4);++k){
// 				int random_block = std::rand()%(MEM_LIMIT-BLK_SZ);
// 				std::memset(threads[i].local_buffer+random_block, 0xAA, BLK_SZ);
// 			}
// 			threads[i].ret = k * BLK_SZ;
// 		});
// 	}
// 	std::size_t total_size = 0;
// 	for(std::size_t i=0;i<N_THREADS;++i){
// 		threads[i].handle->join();
// 		delete[] threads[i].local_buffer;
// 		total_size += threads[i].ret;
// 	}
// 	auto t2 = std::chrono::high_resolution_clock::now();
// 	std::chrono::duration<double, std::milli> time_ra = t2 - t1;
// 	std::cout<<"  "<<time_ra.count()<<"ms @ "<<
// 		static_cast<float>(total_size>>20)/time_ra.count()*1000.f<<"MiB/s"<<std::endl;
}

int main(int argc, char *argv[]){
	init(argc, argv);

	ramdom_write(8 << 20);
	ramdom_write(1 << 20);
	ramdom_write(128 << 10);
	ramdom_write(16 << 10);
	ramdom_write(2 << 10);
	ramdom_write(1 << 10);

	// ramdom_read(512 << 10);
	// ramdom_read(8 << 10);
	// ramdom_read(4 << 10);
	// ramdom_read(2 << 10);
	// ramdom_read(1 << 10);
	// ramdom_read(256);

	return 0;
}
