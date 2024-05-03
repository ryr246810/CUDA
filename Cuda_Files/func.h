#ifndef __FUNC
#define __FUNC

#include <sys/time.h>

// class Enter{
// 	public:
// 		int Add_Gpu();
// };

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif