#ifndef _BaseFunctionDefine_HeaderFile
#define _BaseFunctionDefine_HeaderFile


#include <Standard_TypeDefine.hxx>
#include <set>

#define ALIGNMENT 128

Standard_Integer ThreeDim_DirBump(Standard_Integer dir0, Standard_Integer num);
Standard_Integer TwoDim_DirBump(Standard_Integer dir0, Standard_Integer num);


Standard_Integer CurrIndexOfCircularIndices(const Standard_Integer nb, const Standard_Integer currIndex);
Standard_Integer PrevIndexOfCircularIndices(const Standard_Integer nb, const Standard_Integer currIndex);
Standard_Integer NextIndexOfCircularIndices(const Standard_Integer nb, const Standard_Integer currIndex);

#ifndef __myrdtsc
#define __myrdtsc
static inline unsigned long int getWorkCycle(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" :"=a" (lo),"=d" (hi));
    return ((unsigned long int)hi << 32) | lo;
};
#endif
#define time_chk

#ifndef _Cycle_Def
#define _Cycle_Def
extern Cempic_uint64 cycle_s, cycle_e;
extern Cempic_uint64 worktime[100];
#endif

#ifdef time_chk
#define CYCLE_START (cycle_s = getWorkCycle())
#define CYCLE_CHK(work) \
        do \
        { \
            worktime[work] += getWorkCycle() - cycle_s; \
            cycle_s = getWorkCycle(); \
        } while (0)
                        
#define CYCLE_PRT \
        do \
        { \
        } while (0)
#else
#define CYCLE_START
#define CYCLE_CHK(work)
#define CYCLE_PRT
#endif

bool Bit_Set_BoolOpt_AND(const Standard_Integer theRef, const std::set<Standard_Integer>& theSet);



void v_to_u(const double v, double& u);
void u_to_v(const double u, double& v);


void phi_To_v(const double q, const double m, const double phi, double& v);

void phi_To_u(const double q, const double m, const double phi, double& v);

double energyMKS(double u, double m);

void* aligned_malloc(size_t required_bytes);
void aligned_free(void *p2);

void malloc_Double2D_aligned(double ***head, int X, int Y);
void free_Double2D_aligned(double ***head, int X, int Y);
void malloc_Int2D_aligned(int ***head, int X, int Y);
void free_Int2D_aligned(int ***head, int X, int Y);
void malloc_Double2D_aligned_continuespace(double ***head, int X, int Y);
void free_Double2D_aligned_continuespace(double ***head);
void malloc_Double3D_aligned_continuespace(double ****head, int X, int Y, int Z);
void free_Double3D_aligned_continuespace(double ****head);

void malloc_Double2D(double ***head, int X, int Y);
void free_Double2D(double ***head, int X, int Y);
void malloc_Int2D(int ***head, int X, int Y);
void free_Int2D(int ***head, int X, int Y);

#endif
