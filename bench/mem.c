#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

// clang -Wall -O2 -o build/mem mem.c -lpthread

typedef struct {
	char level; char type;
	size_t size, line, partitions, associativity, sets, threads, cores;
} cache_info_t;

typedef struct {
	pthread_t handle;
	char *local_buffer;
	size_t ret;
} thread_rc;

static size_t N_THREADS;
static size_t MEM_LIMIT = 2ul << 30;
cache_info_t caches[8];

void init(int argc, char *argv[]){
	N_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	for(int i = 0; i < 8; i++) {
		uint32_t eax = 4, ebx, ecx = i, edx;
		asm("cpuid": "+a"(eax), "=b"(ebx), "+c"(ecx), "=d" (edx));
		cache_info_t cache = {
			.type = "$di\x20????????????????????????????"[eax & 0x1F],
			.level = (char)('0' + ((eax >> 5) & 0x7)),
			.threads = ((eax >> 14) & 0xFFF) + 1,
			.cores = ((eax >> 26) & 0x3F) + 1,
			.sets = ecx + 1,
			.line = (ebx & 0xFFF) + 1,
			.partitions = ((ebx >> 12) & 0x3FF) + 1,
			.associativity = ((ebx >> 22) & 0x3FF) + 1,
		};
		cache.size = cache.associativity * cache.partitions * cache.line * cache.sets;
		caches[i] = cache;
	}

#define cache caches[i]
	printf("CPU Cache\n");
	for(int i = 0; i < 8; i++) if (cache.type != '$')
		printf("  L%c%c=%zuKiB (sets%zu assoc%zu part%zu line%zu threads%zu cores%zu)\n",
			cache.level, cache.type, (cache.size>>10),
			cache.sets, cache.associativity, cache.partitions, cache.line,
			cache.threads, cache.cores);
#undef cache

	if(argc > 1){
		int tmp = atoi(argv[1]);
		if(tmp > 0) MEM_LIMIT = (size_t)(tmp) << 20;
	}

	printf("Page size %dB\n", getpagesize());

	MEM_LIMIT /= N_THREADS;
	srand(time(NULL));
}

size_t __BLK_SZ;
void* task_write(void *args){
#define BLK_SZ __BLK_SZ
	thread_rc *_this = (thread_rc*)args;
	size_t k = 0;
	for(;k<((MEM_LIMIT/BLK_SZ)<<4);++k){
		int random_block = rand()%(MEM_LIMIT-BLK_SZ);
		memset(_this->local_buffer+random_block, 0xAA, BLK_SZ);
	}
	_this->ret = k * BLK_SZ;
#undef BLK_SZ
	return NULL;
}

void ramdom_write(size_t BLK_SZ){
	__BLK_SZ = BLK_SZ;
	printf("Writing to %zuMiB block with %zu threads (BLK_SZ = %zuKiB)\n", 
		((MEM_LIMIT*N_THREADS)>>20), N_THREADS, (BLK_SZ>>10));

	thread_rc threads[N_THREADS];
	for(size_t i=0;i<N_THREADS;++i)
		threads[i].local_buffer = (char*)malloc(MEM_LIMIT);

	struct timeval t1;
	gettimeofday(&t1, NULL);
	for(size_t i=0;i<N_THREADS;++i)
		pthread_create(&threads[i].handle, NULL, task_write, &threads[i]);

	size_t total_size = 0;
	for(size_t i=0;i<N_THREADS;++i){
		pthread_join(threads[i].handle, NULL);
		free(threads[i].local_buffer);
		total_size += threads[i].ret;
	}
	struct timeval t2;
	gettimeofday(&t2, NULL);
	uint64_t time_ra = 1000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000;
	printf("  %zums @ %fMiB/s\n", time_ra, ((float)(total_size>>20))/time_ra*1000.f);
}

void* task_read(void *args){
#define BLK_SZ __BLK_SZ
	thread_rc *_this = (thread_rc*)args;
	char private_buffer[BLK_SZ];
	size_t k = 0;
	for(;k<((MEM_LIMIT/BLK_SZ)<<4);++k){
		int random_block = rand()%(MEM_LIMIT-BLK_SZ);
		memcpy(private_buffer, _this->local_buffer+random_block, BLK_SZ);
	}
	_this->ret = k * BLK_SZ;
#undef BLK_SZ
	return NULL;
}

void ramdom_read(size_t BLK_SZ){
	__BLK_SZ = BLK_SZ;
	printf("Reading from %zuMiB block with %zu threads (BLK_SZ = %zuB)\n", 
		((MEM_LIMIT*N_THREADS)>>20), N_THREADS, (BLK_SZ));

	thread_rc threads[N_THREADS];
	for(size_t i=0;i<N_THREADS;++i)
		threads[i].local_buffer = (char*)malloc(MEM_LIMIT);

	struct timeval t1;
	gettimeofday(&t1, NULL);
	for(size_t i=0;i<N_THREADS;++i)
		pthread_create(&threads[i].handle, NULL, task_read, &threads[i]);

	size_t total_size = 0;
	for(size_t i=0;i<N_THREADS;++i){
		pthread_join(threads[i].handle, NULL);
		free(threads[i].local_buffer);
		total_size += threads[i].ret;
	}
	struct timeval t2;
	gettimeofday(&t2, NULL);
	uint64_t time_ra = 1000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000;
	printf("  %zums @ %fGiB/s\n", time_ra, ((float)(total_size>>30))/time_ra*1000.f);
}

int main(int argc, char *argv[]){
	init(argc, argv);

	ramdom_write(8 << 20);
	ramdom_write(1 << 20);
	ramdom_write(128 << 10);
	ramdom_write(16 << 10);
	ramdom_write(2 << 10);
	ramdom_write(1 << 10);

	ramdom_read(512 << 10);
	ramdom_read(8 << 10);
	ramdom_read(4 << 10);
	ramdom_read(2 << 10);
	ramdom_read(1 << 10);
	ramdom_read(256);

	return 0;
}
