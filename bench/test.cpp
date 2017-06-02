#include <boost/thread.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <ctime>

// clang++ -std=c++11 -Wall -O2 -o build/test test.cpp -lstdc++ -lboost_thread -lpthread -lboost_filesystem -lboost_system

typedef struct {
	char level; char type;
	std::size_t size, line, partitions, associativity, sets, threads, cores;
} cache_info_t;

static std::size_t N_THREADS;
static std::size_t MEM_LIMIT = 2ul << 30;
static std::size_t BLK_SZ = 8 << 20;
std::vector<cache_info_t> caches;

void cpuid(std::vector<cache_info_t> &output){
	for(int i = 0; i < 8; i++) {
		std::uint32_t eax = 4, ebx, ecx = i, edx; 
		asm("cpuid": "+a"(eax), "=b"(ebx), "+c"(ecx), "=d" (edx));
		cache_info_t cache = {
			.type = "$di\x20????????????????????????????"[eax & 0x1F],
			.level = static_cast<char>('0' + ((eax >> 5) & 0x7)),
			.threads = ((eax >> 14) & 0xFFF) + 1,
			.cores = ((eax >> 26) & 0x3F) + 1,
			.sets = ecx + 1,
			.line = (ebx & 0xFFF) + 1,
			.partitions = ((ebx >> 12) & 0x3FF) + 1,
			.associativity = ((ebx >> 22) & 0x3FF) + 1,
		};
		if(cache.type == '$') break;
		cache.size = cache.associativity * cache.partitions * cache.line * cache.sets;
		output.push_back(cache);
	}
}

void init(int argc, char *argv[]){
	N_THREADS = boost::thread::hardware_concurrency();
	cpuid(caches);

	for(auto const& cache: caches)
		std::cout<<'L'<<cache.level<<cache.type<<'='<<(cache.size>>10)<<
			"KiB (sets"<<cache.sets<<" assoc"<<cache.associativity<<" part"<<cache.partitions<<" line"<<cache.line<<
			" threads"<<cache.threads<<" cores"<<cache.cores<<')'<<std::endl;

	if(argc > 1){
		std::cout<<sizeof(std::size_t)<<std::endl;
		int tmp = atoi(argv[1]);
		if(tmp > 0) MEM_LIMIT = static_cast<std::size_t>(tmp) << 20;
	}

	MEM_LIMIT /= N_THREADS;
}

int main(int argc, char *argv[]){
	init(argc, argv);

	std::cout<<"Writing to "<<((MEM_LIMIT*N_THREADS)>>20)<<"MiB block with "<<
		N_THREADS<<" threads (BLK_SZ = "<<(BLK_SZ>>20)<<"MiB)..."<<std::endl;

	char **buffer = new char*[N_THREADS];
	for(std::size_t i=0;i<N_THREADS;++i)
		buffer[i] = new char[MEM_LIMIT];
	int const *BLOCK = new int[BLK_SZ];
	std::srand(std::time(0));

	boost::thread *children[N_THREADS];
	for(std::size_t i=0;i<N_THREADS;++i){
		children[i] = new boost::thread([&,i](){
			for(int k=0;k<((MEM_LIMIT/BLK_SZ)<<5);++k){
				int random_block = std::rand()%(MEM_LIMIT-BLK_SZ);
				std::memcpy(buffer[i]+random_block, BLOCK, BLK_SZ);
			}
		});
	}
	for(std::size_t i=0;i<N_THREADS;++i)
		children[i]->join();
	std::cout<<"Randam access"<<std::endl;
	return 0;
}
