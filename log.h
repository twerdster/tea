#ifndef __LOG_H__
#define __LOG_H__

#include <boost/thread.hpp>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cstdio>

static boost::mutex mtx;
inline std::string NowTime();

enum TLogLevel {logERROR, logWARNING, logINFO, LOG0, LOG1, LOG2, LOG3, LOG4};

template <typename T>
class Log
{
public:
    Log();
    virtual ~Log();
    std::ostringstream& Get(TLogLevel level = logINFO);
public:
    static TLogLevel& ReportingLevel();
    static std::string ToString(TLogLevel level);
    static TLogLevel FromString(const std::string& level);
protected:
    std::ostringstream os;
private:
    Log(const Log&);
    Log& operator =(const Log&);
};

template <typename T>
Log<T>::Log(){}

template <typename T>
std::ostringstream& Log<T>::Get(TLogLevel level)
{
    os << "- " << NowTime();
    os << " " << ToString(level) << ": ";
    os << std::string(level > LOG0 ? level - LOG0 : 0, '\t');
	os.precision(4);
    return os;
}

template <typename T>
Log<T>::~Log()
{
    os << std::endl;
    T::Output(os.str());
}

template <typename T>
TLogLevel& Log<T>::ReportingLevel()
{
    static TLogLevel reportingLevel = LOG4;
    return reportingLevel;
}

template <typename T>
std::string Log<T>::ToString(TLogLevel level)
{
	static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "LOG0", "LOG1", "LOG2", "LOG3", "LOG4"};
    return buffer[level];
}

template <typename T>
TLogLevel Log<T>::FromString(const std::string& level)
{
    if (level == "LOG4")        return LOG4;
    if (level == "LOG3")        return LOG3;
    if (level == "LOG2")        return LOG2;
    if (level == "LOG1")        return LOG1;
    if (level == "LOG0")        return LOG0;
    if (level == "INFO")        return logINFO;
    if (level == "WARNING")        return logWARNING;
    if (level == "ERROR")        return logERROR;

    Log<T>().Get(logWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
    return logINFO;
}


class Output2FILE
{
public:
    static void Output(const std::string& msg);
    static void SetStream(FILE* pFile);	
    static FILE*& Stream();
};

inline FILE*& Output2FILE::Stream()
{
    static FILE* pStream = 0;
    return pStream;
}

inline void Output2FILE::SetStream(FILE* pFile)
{
    boost::mutex::scoped_lock lock(mtx);
    Stream() = pFile;
}

inline void Output2FILE::Output(const std::string& msg)
{
    boost::mutex::scoped_lock lock(mtx);
    FILE* pStream = Stream();
	fprintf(stderr, "%s", msg.c_str());
    fflush(stderr);
    if (!pStream)
        return;
    fprintf(pStream, "%s", msg.c_str());
    fflush(pStream);
}

class FILELog : public Log<Output2FILE> {};
//typedef Log<Output2FILE> FILELog;

#ifndef FILELOG_MAX_LEVEL
#define FILELOG_MAX_LEVEL LOG4
#endif

#define FILE_LOG(level) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel() ) ; \
    else FILELog().Get(level)

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)

#include <windows.h>

inline std::string NowTime()
{
    const int MAX_LEN = 200;
    char buffer[MAX_LEN];
    if (GetTimeFormatA(LOCALE_USER_DEFAULT, 0, 0, 
            "HH':'mm':'ss", buffer, MAX_LEN) == 0)
        return "Error in NowTime()";

    char result[100] = {0};
    static DWORD first = GetTickCount();
    std::sprintf(result, "%s.%03ld", buffer, (long)(GetTickCount() - first) % 1000); 
    return result;
}

#else

#include <sys/time.h>

inline std::string NowTime()
{
    char buffer[11];
    time_t t;
    time(&t);
    tm r = {0};
    strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));
    struct timeval tv;
    gettimeofday(&tv, 0);
    char result[100] = {0};
    std::sprintf(result, "%s.%03ld", buffer, (long)tv.tv_usec / 1000); 
    return result;
}

#endif //WIN32

#endif //__LOG_H__
