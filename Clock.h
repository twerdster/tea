#ifndef CLOCK
#define CLOCK


/*
#include <iostream>
#include <chrono>
class Clock
{
public:
	Clock() {}
	inline void tic()  {start_ = std::chrono::high_resolution_clock::now();}
	inline std::chrono::nanoseconds::rep toc()  {
		end_ = std::chrono::high_resolution_clock::now(); 
		last_=std::chrono::duration_cast<std::chrono::nanoseconds>(end_-start_).count(); return last_;}
	inline std::chrono::nanoseconds::rep last() {return last_;}
private:
	std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_; 
    std::chrono::nanoseconds::rep last_;
};
*/

class Clock
{
public:
	Clock() {}
	inline void tic()  {}
	inline double toc()  {return -1;}
	inline double last() {return -1;}
};

#endif