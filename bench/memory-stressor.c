#define _BSD_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_FOOTPRINT	(1024)		// 1GB
#define MALLOC_SIZE			(1024*1024)	// 1MB

int main(int argc, char** argv) 
{
	int footprint_mb = DEFAULT_FOOTPRINT;
	if(argc > 1) {
		footprint_mb = atoi(argv[1]);
	}

	nice(10);
	
	printf("Pre-allocating %dMB...\n", footprint_mb);

	char **buffers = malloc(sizeof(char *)*footprint_mb);
	for (int i=0 ; i<footprint_mb ; i++) {
		buffers[i] = malloc(MALLOC_SIZE);
	}
	printf("Done. Starting to touch that memory randomly... press ctrl+C to exit.\n");

	srand(time(NULL));
	int iter = 0;
	for (;;) {
		int buffer_num = rand() % footprint_mb;
		memset(buffers[buffer_num], 0, MALLOC_SIZE);
		if (++iter >= 64) {
			printf("Wrote buffer %5d (and %d others); %dMB total touched.\n", buffer_num, iter-1, iter);
			sleep(1);
			iter = 0;
		}
	}

	return 0;
}
