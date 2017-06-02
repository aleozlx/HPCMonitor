#include <boost/thread.hpp>
#include <iostream>
// clang++ -std=c++11 -Wall -O2 -o build/test test.cpp -lstdc++ -lboost_thread -lpthread -lboost_filesystem -lboost_system
int main(){
	unsigned int nthreads = boost::thread::hardware_concurrency();
	boost::thread *children[nthreads];
	for(std::size_t i=0;i<nthreads;++i){
		children[i] = new boost::thread([&,i](){
            std::cout<<i<<std::endl;
        });
	}
	for(std::size_t i=0;i<nthreads;++i)
		children[i]->join();
	return 0;
}
