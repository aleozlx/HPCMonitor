#include <boost/thread.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

// clang++ -std=c++11 -Wall -O2 -o build/test test.cpp -lstdc++ -lboost_thread -lpthread -lboost_filesystem -lboost_system

static std::size_t MEM_LIMIT = 2ul << 30;
static const std::size_t BLK_SZ = 1 << 20;

int main(int argc, char *argv[]){
	unsigned int N_THREADS = boost::thread::hardware_concurrency();

	if(argc > 1){
		std::cout<<sizeof(std::size_t)<<std::endl;
		int tmp = atoi(argv[1]);
		if(tmp > 0) MEM_LIMIT = static_cast<std::size_t>(tmp) << 20;
	}

	std::cout<<"Writing to "<<(MEM_LIMIT>>20)<<"MiB block with "<<N_THREADS<<" threads..."<<std::endl;

	MEM_LIMIT /= N_THREADS;
	char **buffer = new char*[N_THREADS];
	for(std::size_t i=0;i<N_THREADS;++i)
		buffer[i] = new char[MEM_LIMIT];
	int const *BLOCK = new int[1 << 20];
	std::srand(std::time(0));

	boost::thread *children[N_THREADS];
	for(std::size_t i=0;i<N_THREADS;++i){
		children[i] = new boost::thread([&,i](){
			for(int k=0;k<((MEM_LIMIT/BLK_SZ)<<5);++k){
				int random_block = std::rand();
				if(random_block > (MEM_LIMIT-BLK_SZ)) continue;
				std::memcpy(buffer[i]+random_block, BLOCK, BLK_SZ);
			}
		});
	}
	for(std::size_t i=0;i<N_THREADS;++i)
		children[i]->join();
	
	return 0;
}
