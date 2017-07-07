#ifndef CLOCK
#define CLOCK



#include <iostream>
#include <chrono>
class Clock
{
public:
	Clock() {freq_=1000.0f*1000.0f*1000.0f;}
	inline void tic()  {start_ = std::chrono::high_resolution_clock::now();}
	inline double toc()  {
		end_ = std::chrono::high_resolution_clock::now(); 
		last_=std::chrono::duration_cast<std::chrono::nanoseconds>(end_-start_).count(); return last_/freq_;}
	inline double last() {return last_/freq_;}
private:
	double freq_;
	std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_; 
    std::chrono::nanoseconds::rep last_;
};

/*
class Clock
{
public:
	Clock() {}
	inline void tic()  {}
	inline double toc()  {return -1;}
	inline double last() {return -1;}
};
*/
#endif