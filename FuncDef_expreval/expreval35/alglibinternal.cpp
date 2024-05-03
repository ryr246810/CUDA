/*************************************************************************
ALGLIB 3.14.0 (source code generated 2018-06-16)
Copyright (c) Sergey Bochkanov (ALGLIB project).

>>> SOURCE LICENSE >>>
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation (www.fsf.org); either version 2 of the 
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available at
http://www.fsf.org/licensing/licenses
>>> END OF LICENSE >>>
*************************************************************************/
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "alglibinternal.h"

// disable some irrelevant warnings
#if (AE_COMPILER==AE_MSVC) && !defined(AE_ALL_WARNINGS)
// #pragma warning(disable:4100)
// #pragma warning(disable:4127)
// #pragma warning(disable:4611)
// #pragma warning(disable:4702)
// #pragma warning(disable:4996)
#endif

/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS IMPLEMENTATION OF C++ INTERFACE
//
/////////////////////////////////////////////////////////////////////////
namespace alglib
{


}

/////////////////////////////////////////////////////////////////////////
//
// THIS SECTION CONTAINS IMPLEMENTATION OF COMPUTATIONAL CORE
//
/////////////////////////////////////////////////////////////////////////
namespace alglib_impl
{
#if defined(AE_COMPILE_SCODES) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_APSERV) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_TSORT) || !defined(AE_PARTIAL_BUILD)
static void tsort_tagsortfastirec(/* Real    */ ae_vector* a,
     /* Integer */ ae_vector* b,
     /* Real    */ ae_vector* bufa,
     /* Integer */ ae_vector* bufb,
     ae_int_t i1,
     ae_int_t i2,
     ae_state *_state);
static void tsort_tagsortfastrrec(/* Real    */ ae_vector* a,
     /* Real    */ ae_vector* b,
     /* Real    */ ae_vector* bufa,
     /* Real    */ ae_vector* bufb,
     ae_int_t i1,
     ae_int_t i2,
     ae_state *_state);
static void tsort_tagsortfastrec(/* Real    */ ae_vector* a,
     /* Real    */ ae_vector* bufa,
     ae_int_t i1,
     ae_int_t i2,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_ABLASMKL) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_ABLASF) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_CREFLECTIONS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_ROTATIONS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_TRLINSOLVE) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_SAFESOLVE) || !defined(AE_PARTIAL_BUILD)
static ae_bool safesolve_cbasicsolveandupdate(ae_complex alpha,
     ae_complex beta,
     double lnmax,
     double bnorm,
     double maxgrowth,
     double* xnorm,
     ae_complex* x,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_HBLAS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_SBLAS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_BLAS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_LINMIN) || !defined(AE_PARTIAL_BUILD)
static double linmin_ftol = 0.001;
static double linmin_xtol = 100*ae_machineepsilon;
static ae_int_t linmin_maxfev = 20;
static double linmin_stpmin = 1.0E-50;
static double linmin_defstpmax = 1.0E+50;
static double linmin_armijofactor = 1.3;
static void linmin_mcstep(double* stx,
     double* fx,
     double* dx,
     double* sty,
     double* fy,
     double* dy,
     double* stp,
     double fp,
     double dp,
     ae_bool* brackt,
     double stmin,
     double stmax,
     ae_int_t* info,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_XBLAS) || !defined(AE_PARTIAL_BUILD)
static void xblas_xsum(/* Real    */ ae_vector* w,
     double mx,
     ae_int_t n,
     double* r,
     double* rerr,
     ae_state *_state);
static double xblas_xfastpow(double r, ae_int_t n, ae_state *_state);


#endif
#if defined(AE_COMPILE_BASICSTATOPS) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_HPCCORES) || !defined(AE_PARTIAL_BUILD)
static ae_bool hpccores_hpcpreparechunkedgradientx(/* Real    */ ae_vector* weights,
     ae_int_t wcount,
     /* Real    */ ae_vector* hpcbuf,
     ae_state *_state);
static ae_bool hpccores_hpcfinalizechunkedgradientx(/* Real    */ ae_vector* buf,
     ae_int_t wcount,
     /* Real    */ ae_vector* grad,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_NTHEORY) || !defined(AE_PARTIAL_BUILD)
static ae_bool ntheory_isprime(ae_int_t n, ae_state *_state);
static ae_int_t ntheory_modmul(ae_int_t a,
     ae_int_t b,
     ae_int_t n,
     ae_state *_state);
static ae_int_t ntheory_modexp(ae_int_t a,
     ae_int_t b,
     ae_int_t n,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_FTBASE) || !defined(AE_PARTIAL_BUILD)
static ae_int_t ftbase_coltype = 0;
static ae_int_t ftbase_coloperandscnt = 1;
static ae_int_t ftbase_coloperandsize = 2;
static ae_int_t ftbase_colmicrovectorsize = 3;
static ae_int_t ftbase_colparam0 = 4;
static ae_int_t ftbase_colparam1 = 5;
static ae_int_t ftbase_colparam2 = 6;
static ae_int_t ftbase_colparam3 = 7;
static ae_int_t ftbase_colscnt = 8;
static ae_int_t ftbase_opend = 0;
static ae_int_t ftbase_opcomplexreffft = 1;
static ae_int_t ftbase_opbluesteinsfft = 2;
static ae_int_t ftbase_opcomplexcodeletfft = 3;
static ae_int_t ftbase_opcomplexcodelettwfft = 4;
static ae_int_t ftbase_opradersfft = 5;
static ae_int_t ftbase_opcomplextranspose = -1;
static ae_int_t ftbase_opcomplexfftfactors = -2;
static ae_int_t ftbase_opstart = -3;
static ae_int_t ftbase_opjmp = -4;
static ae_int_t ftbase_opparallelcall = -5;
static ae_int_t ftbase_maxradix = 6;
static ae_int_t ftbase_updatetw = 16;
static ae_int_t ftbase_recursivethreshold = 1024;
static ae_int_t ftbase_raderthreshold = 19;
static ae_int_t ftbase_ftbasecodeletrecommended = 5;
static double ftbase_ftbaseinefficiencyfactor = 1.3;
static ae_int_t ftbase_ftbasemaxsmoothfactor = 5;
static void ftbase_ftdeterminespacerequirements(ae_int_t n,
     ae_int_t* precrsize,
     ae_int_t* precisize,
     ae_state *_state);
static void ftbase_ftcomplexfftplanrec(ae_int_t n,
     ae_int_t k,
     ae_bool childplan,
     ae_bool topmostplan,
     ae_int_t* rowptr,
     ae_int_t* bluesteinsize,
     ae_int_t* precrptr,
     ae_int_t* preciptr,
     fasttransformplan* plan,
     ae_state *_state);
static void ftbase_ftpushentry(fasttransformplan* plan,
     ae_int_t* rowptr,
     ae_int_t etype,
     ae_int_t eopcnt,
     ae_int_t eopsize,
     ae_int_t emcvsize,
     ae_int_t eparam0,
     ae_state *_state);
static void ftbase_ftpushentry2(fasttransformplan* plan,
     ae_int_t* rowptr,
     ae_int_t etype,
     ae_int_t eopcnt,
     ae_int_t eopsize,
     ae_int_t emcvsize,
     ae_int_t eparam0,
     ae_int_t eparam1,
     ae_state *_state);
static void ftbase_ftpushentry4(fasttransformplan* plan,
     ae_int_t* rowptr,
     ae_int_t etype,
     ae_int_t eopcnt,
     ae_int_t eopsize,
     ae_int_t emcvsize,
     ae_int_t eparam0,
     ae_int_t eparam1,
     ae_int_t eparam2,
     ae_int_t eparam3,
     ae_state *_state);
static void ftbase_ftapplysubplan(fasttransformplan* plan,
     ae_int_t subplan,
     /* Real    */ ae_vector* a,
     ae_int_t abase,
     ae_int_t aoffset,
     /* Real    */ ae_vector* buf,
     ae_int_t repcnt,
     ae_state *_state);
static void ftbase_ftapplycomplexreffft(/* Real    */ ae_vector* a,
     ae_int_t offs,
     ae_int_t operandscnt,
     ae_int_t operandsize,
     ae_int_t microvectorsize,
     /* Real    */ ae_vector* buf,
     ae_state *_state);
static void ftbase_ftapplycomplexcodeletfft(/* Real    */ ae_vector* a,
     ae_int_t offs,
     ae_int_t operandscnt,
     ae_int_t operandsize,
     ae_int_t microvectorsize,
     ae_state *_state);
static void ftbase_ftapplycomplexcodelettwfft(/* Real    */ ae_vector* a,
     ae_int_t offs,
     ae_int_t operandscnt,
     ae_int_t operandsize,
     ae_int_t microvectorsize,
     ae_state *_state);
static void ftbase_ftprecomputebluesteinsfft(ae_int_t n,
     ae_int_t m,
     /* Real    */ ae_vector* precr,
     ae_int_t offs,
     ae_state *_state);
static void ftbase_ftbluesteinsfft(fasttransformplan* plan,
     /* Real    */ ae_vector* a,
     ae_int_t abase,
     ae_int_t aoffset,
     ae_int_t operandscnt,
     ae_int_t n,
     ae_int_t m,
     ae_int_t precoffs,
     ae_int_t subplan,
     /* Real    */ ae_vector* bufa,
     /* Real    */ ae_vector* bufb,
     /* Real    */ ae_vector* bufc,
     /* Real    */ ae_vector* bufd,
     ae_state *_state);
static void ftbase_ftprecomputeradersfft(ae_int_t n,
     ae_int_t rq,
     ae_int_t riq,
     /* Real    */ ae_vector* precr,
     ae_int_t offs,
     ae_state *_state);
static void ftbase_ftradersfft(fasttransformplan* plan,
     /* Real    */ ae_vector* a,
     ae_int_t abase,
     ae_int_t aoffset,
     ae_int_t operandscnt,
     ae_int_t n,
     ae_int_t subplan,
     ae_int_t rq,
     ae_int_t riq,
     ae_int_t precoffs,
     /* Real    */ ae_vector* buf,
     ae_state *_state);
static void ftbase_ftfactorize(ae_int_t n,
     ae_bool isroot,
     ae_int_t* n1,
     ae_int_t* n2,
     ae_state *_state);
static ae_int_t ftbase_ftoptimisticestimate(ae_int_t n, ae_state *_state);
static void ftbase_ffttwcalc(/* Real    */ ae_vector* a,
     ae_int_t aoffset,
     ae_int_t n1,
     ae_int_t n2,
     ae_state *_state);
static void ftbase_internalcomplexlintranspose(/* Real    */ ae_vector* a,
     ae_int_t m,
     ae_int_t n,
     ae_int_t astart,
     /* Real    */ ae_vector* buf,
     ae_state *_state);
static void ftbase_ffticltrec(/* Real    */ ae_vector* a,
     ae_int_t astart,
     ae_int_t astride,
     /* Real    */ ae_vector* b,
     ae_int_t bstart,
     ae_int_t bstride,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state);
static void ftbase_fftirltrec(/* Real    */ ae_vector* a,
     ae_int_t astart,
     ae_int_t astride,
     /* Real    */ ae_vector* b,
     ae_int_t bstart,
     ae_int_t bstride,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state);
static void ftbase_ftbasefindsmoothrec(ae_int_t n,
     ae_int_t seed,
     ae_int_t leastfactor,
     ae_int_t* best,
     ae_state *_state);


#endif
#if defined(AE_COMPILE_NEARUNITYUNIT) || !defined(AE_PARTIAL_BUILD)


#endif
#if defined(AE_COMPILE_ALGLIBBASICS) || !defined(AE_PARTIAL_BUILD)


#endif

#if defined(AE_COMPILE_SCODES) || !defined(AE_PARTIAL_BUILD)


ae_int_t getrdfserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 1;
    return result;
}


ae_int_t getkdtreeserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 2;
    return result;
}


ae_int_t getmlpserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 3;
    return result;
}


ae_int_t getmlpeserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 4;
    return result;
}


ae_int_t getrbfserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 5;
    return result;
}


ae_int_t getspline2dserializationcode(ae_state *_state)
{
    ae_int_t result;


    result = 6;
    return result;
}


#endif
#if defined(AE_COMPILE_APSERV) || !defined(AE_PARTIAL_BUILD)


/*************************************************************************
Internally calls SetErrorFlag() with condition:

    Abs(Val-RefVal)>Tol*Max(Abs(RefVal),S)
    
This function is used to test relative error in Val against  RefVal,  with
relative error being replaced by absolute when scale  of  RefVal  is  less
than S.

This function returns value of COND.
*************************************************************************/
void seterrorflagdiff(ae_bool* flag,
     double val,
     double refval,
     double tol,
     double s,
     ae_state *_state)
{


    ae_set_error_flag(flag, ae_fp_greater(ae_fabs(val-refval, _state),tol*ae_maxreal(ae_fabs(refval, _state), s, _state)), __FILE__, __LINE__, "apserv.ap:162");
}


/*************************************************************************
The function always returns False.
It may be used sometimes to prevent spurious warnings.

  -- ALGLIB --
     Copyright 17.09.2012 by Bochkanov Sergey
*************************************************************************/
ae_bool alwaysfalse(ae_state *_state)
{
    ae_bool result;


    result = ae_false;
    return result;
}


/*************************************************************************
The function "touches" integer - it is used  to  avoid  compiler  messages
about unused variables (in rare cases when we do NOT want to remove  these
variables).

  -- ALGLIB --
     Copyright 17.09.2012 by Bochkanov Sergey
*************************************************************************/
void touchint(ae_int_t* a, ae_state *_state)
{


}


/*************************************************************************
The function "touches" real   -  it is used  to  avoid  compiler  messages
about unused variables (in rare cases when we do NOT want to remove  these
variables).

  -- ALGLIB --
     Copyright 17.09.2012 by Bochkanov Sergey
*************************************************************************/
void touchreal(double* a, ae_state *_state)
{


}


/*************************************************************************
The function performs zero-coalescing on real value.

NOTE: no check is performed for B<>0

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
double coalesce(double a, double b, ae_state *_state)
{
    double result;


    result = a;
    if( ae_fp_eq(a,0.0) )
    {
        result = b;
    }
    return result;
}


/*************************************************************************
The function performs zero-coalescing on integer value.

NOTE: no check is performed for B<>0

  -- ALGLIB --
     Copyright 18.05.2015 by Bochkanov Sergey
*************************************************************************/
ae_int_t coalescei(ae_int_t a, ae_int_t b, ae_state *_state)
{
    ae_int_t result;


    result = a;
    if( a==0 )
    {
        result = b;
    }
    return result;
}


/*************************************************************************
The function convert integer value to real value.

  -- ALGLIB --
     Copyright 17.09.2012 by Bochkanov Sergey
*************************************************************************/
double inttoreal(ae_int_t a, ae_state *_state)
{
    double result;


    result = (double)(a);
    return result;
}


/*************************************************************************
The function calculates binary logarithm.

NOTE: it costs twice as much as Ln(x)

  -- ALGLIB --
     Copyright 17.09.2012 by Bochkanov Sergey
*************************************************************************/
double logbase2(double x, ae_state *_state)
{
    double result;


    result = ae_log(x, _state)/ae_log((double)(2), _state);
    return result;
}


/*************************************************************************
This function compares two numbers for approximate equality, with tolerance
to errors as large as tol.


  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
ae_bool approxequal(double a, double b, double tol, ae_state *_state)
{
    ae_bool result;


    result = ae_fp_less_eq(ae_fabs(a-b, _state),tol);
    return result;
}


/*************************************************************************
This function compares two numbers for approximate equality, with tolerance
to errors as large as max(|a|,|b|)*tol.


  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
ae_bool approxequalrel(double a, double b, double tol, ae_state *_state)
{
    ae_bool result;


    result = ae_fp_less_eq(ae_fabs(a-b, _state),ae_maxreal(ae_fabs(a, _state), ae_fabs(b, _state), _state)*tol);
    return result;
}


/*************************************************************************
This  function  generates  1-dimensional  general  interpolation task with
moderate Lipshitz constant (close to 1.0)

If N=1 then suborutine generates only one point at the middle of [A,B]

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
void taskgenint1d(double a,
     double b,
     ae_int_t n,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* y,
     ae_state *_state)
{
    ae_int_t i;
    double h;

    ae_vector_clear(x);
    ae_vector_clear(y);

    ae_assert(n>=1, "TaskGenInterpolationEqdist1D: N<1!", _state);
    ae_vector_set_length(x, n, _state);
    ae_vector_set_length(y, n, _state);
    if( n>1 )
    {
        x->ptr.p_double[0] = a;
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
        h = (b-a)/(n-1);
        for(i=1; i<=n-1; i++)
        {
            if( i!=n-1 )
            {
                x->ptr.p_double[i] = a+(i+0.2*(2*ae_randomreal(_state)-1))*h;
            }
            else
            {
                x->ptr.p_double[i] = b;
            }
            y->ptr.p_double[i] = y->ptr.p_double[i-1]+(2*ae_randomreal(_state)-1)*(x->ptr.p_double[i]-x->ptr.p_double[i-1]);
        }
    }
    else
    {
        x->ptr.p_double[0] = 0.5*(a+b);
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
    }
}


/*************************************************************************
This function generates  1-dimensional equidistant interpolation task with
moderate Lipshitz constant (close to 1.0)

If N=1 then suborutine generates only one point at the middle of [A,B]

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
void taskgenint1dequidist(double a,
     double b,
     ae_int_t n,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* y,
     ae_state *_state)
{
    ae_int_t i;
    double h;

    ae_vector_clear(x);
    ae_vector_clear(y);

    ae_assert(n>=1, "TaskGenInterpolationEqdist1D: N<1!", _state);
    ae_vector_set_length(x, n, _state);
    ae_vector_set_length(y, n, _state);
    if( n>1 )
    {
        x->ptr.p_double[0] = a;
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
        h = (b-a)/(n-1);
        for(i=1; i<=n-1; i++)
        {
            x->ptr.p_double[i] = a+i*h;
            y->ptr.p_double[i] = y->ptr.p_double[i-1]+(2*ae_randomreal(_state)-1)*h;
        }
    }
    else
    {
        x->ptr.p_double[0] = 0.5*(a+b);
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
    }
}


/*************************************************************************
This function generates  1-dimensional Chebyshev-1 interpolation task with
moderate Lipshitz constant (close to 1.0)

If N=1 then suborutine generates only one point at the middle of [A,B]

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
void taskgenint1dcheb1(double a,
     double b,
     ae_int_t n,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* y,
     ae_state *_state)
{
    ae_int_t i;

    ae_vector_clear(x);
    ae_vector_clear(y);

    ae_assert(n>=1, "TaskGenInterpolation1DCheb1: N<1!", _state);
    ae_vector_set_length(x, n, _state);
    ae_vector_set_length(y, n, _state);
    if( n>1 )
    {
        for(i=0; i<=n-1; i++)
        {
            x->ptr.p_double[i] = 0.5*(b+a)+0.5*(b-a)*ae_cos(ae_pi*(2*i+1)/(2*n), _state);
            if( i==0 )
            {
                y->ptr.p_double[i] = 2*ae_randomreal(_state)-1;
            }
            else
            {
                y->ptr.p_double[i] = y->ptr.p_double[i-1]+(2*ae_randomreal(_state)-1)*(x->ptr.p_double[i]-x->ptr.p_double[i-1]);
            }
        }
    }
    else
    {
        x->ptr.p_double[0] = 0.5*(a+b);
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
    }
}


/*************************************************************************
This function generates  1-dimensional Chebyshev-2 interpolation task with
moderate Lipshitz constant (close to 1.0)

If N=1 then suborutine generates only one point at the middle of [A,B]

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
void taskgenint1dcheb2(double a,
     double b,
     ae_int_t n,
     /* Real    */ ae_vector* x,
     /* Real    */ ae_vector* y,
     ae_state *_state)
{
    ae_int_t i;

    ae_vector_clear(x);
    ae_vector_clear(y);

    ae_assert(n>=1, "TaskGenInterpolation1DCheb2: N<1!", _state);
    ae_vector_set_length(x, n, _state);
    ae_vector_set_length(y, n, _state);
    if( n>1 )
    {
        for(i=0; i<=n-1; i++)
        {
            x->ptr.p_double[i] = 0.5*(b+a)+0.5*(b-a)*ae_cos(ae_pi*i/(n-1), _state);
            if( i==0 )
            {
                y->ptr.p_double[i] = 2*ae_randomreal(_state)-1;
            }
            else
            {
                y->ptr.p_double[i] = y->ptr.p_double[i-1]+(2*ae_randomreal(_state)-1)*(x->ptr.p_double[i]-x->ptr.p_double[i-1]);
            }
        }
    }
    else
    {
        x->ptr.p_double[0] = 0.5*(a+b);
        y->ptr.p_double[0] = 2*ae_randomreal(_state)-1;
    }
}


/*************************************************************************
This function checks that all values from X[] are distinct. It does more
than just usual floating point comparison:
* first, it calculates max(X) and min(X)
* second, it maps X[] from [min,max] to [1,2]
* only at this stage actual comparison is done

The meaning of such check is to ensure that all values are "distinct enough"
and will not cause interpolation subroutine to fail.

NOTE:
    X[] must be sorted by ascending (subroutine ASSERT's it)

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
ae_bool aredistinct(/* Real    */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    double a;
    double b;
    ae_int_t i;
    ae_bool nonsorted;
    ae_bool result;


    ae_assert(n>=1, "APSERVAreDistinct: internal error (N<1)", _state);
    if( n==1 )
    {
        
        /*
         * everything is alright, it is up to caller to decide whether it
         * can interpolate something with just one point
         */
        result = ae_true;
        return result;
    }
    a = x->ptr.p_double[0];
    b = x->ptr.p_double[0];
    nonsorted = ae_false;
    for(i=1; i<=n-1; i++)
    {
        a = ae_minreal(a, x->ptr.p_double[i], _state);
        b = ae_maxreal(b, x->ptr.p_double[i], _state);
        nonsorted = nonsorted||ae_fp_greater_eq(x->ptr.p_double[i-1],x->ptr.p_double[i]);
    }
    ae_assert(!nonsorted, "APSERVAreDistinct: internal error (not sorted)", _state);
    for(i=1; i<=n-1; i++)
    {
        if( ae_fp_eq((x->ptr.p_double[i]-a)/(b-a)+1,(x->ptr.p_double[i-1]-a)/(b-a)+1) )
        {
            result = ae_false;
            return result;
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that two boolean values are the same (both  are  True 
or both are False).

  -- ALGLIB --
     Copyright 02.12.2009 by Bochkanov Sergey
*************************************************************************/
ae_bool aresameboolean(ae_bool v1, ae_bool v2, ae_state *_state)
{
    ae_bool result;


    result = (v1&&v2)||(!v1&&!v2);
    return result;
}


/*************************************************************************
If Length(X)<N, resizes X

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void bvectorsetlengthatleast(/* Boolean */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{


    if( x->cnt<n )
    {
        ae_vector_set_length(x, n, _state);
    }
}


/*************************************************************************
If Length(X)<N, resizes X

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void ivectorsetlengthatleast(/* Integer */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{


    if( x->cnt<n )
    {
        ae_vector_set_length(x, n, _state);
    }
}


/*************************************************************************
If Length(X)<N, resizes X

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rvectorsetlengthatleast(/* Real    */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{


    if( x->cnt<n )
    {
        ae_vector_set_length(x, n, _state);
    }
}


/*************************************************************************
If Cols(X)<N or Rows(X)<M, resizes X

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rmatrixsetlengthatleast(/* Real    */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{


    if( m>0&&n>0 )
    {
        if( x->rows<m||x->cols<n )
        {
            ae_matrix_set_length(x, m, n, _state);
        }
    }
}


/*************************************************************************
Grows X, i.e. changes its size in such a way that:
a) contents is preserved
b) new size is at least N
c) new size can be larger than N, so subsequent grow() calls can return
   without reallocation

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void ivectorgrowto(/* Integer */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_vector oldx;
    ae_int_t i;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_vector_init(&oldx, 0, DT_INT, _state, ae_true);

    
    /*
     * Enough place
     */
    if( x->cnt>=n )
    {
        ae_frame_leave(_state);
        return;
    }
    
    /*
     * Choose new size
     */
    n = ae_maxint(n, ae_round(1.8*x->cnt+1, _state), _state);
    
    /*
     * Grow
     */
    n2 = x->cnt;
    ae_swap_vectors(x, &oldx);
    ae_vector_set_length(x, n, _state);
    for(i=0; i<=n-1; i++)
    {
        if( i<n2 )
        {
            x->ptr.p_int[i] = oldx.ptr.p_int[i];
        }
        else
        {
            x->ptr.p_int[i] = 0;
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Grows X, i.e. appends rows in such a way that:
a) contents is preserved
b) new row count is at least N
c) new row count can be larger than N, so subsequent grow() calls can return
   without reallocation
d) new matrix has at least MinCols columns (if less than specified amount
   of columns is present, new columns are added with undefined contents);
   MinCols can be 0 or negative value = ignored

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rmatrixgrowrowsto(/* Real    */ ae_matrix* a,
     ae_int_t n,
     ae_int_t mincols,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_matrix olda;
    ae_int_t i;
    ae_int_t j;
    ae_int_t n2;
    ae_int_t m;

    ae_frame_make(_state, &_frame_block);
    memset(&olda, 0, sizeof(olda));
    ae_matrix_init(&olda, 0, 0, DT_REAL, _state, ae_true);

    
    /*
     * Enough place?
     */
    if( a->rows>=n&&a->cols>=mincols )
    {
        ae_frame_leave(_state);
        return;
    }
    
    /*
     * Sizes and metrics
     */
    if( a->rows<n )
    {
        n = ae_maxint(n, ae_round(1.8*a->rows+1, _state), _state);
    }
    n2 = ae_minint(a->rows, n, _state);
    m = a->cols;
    
    /*
     * Grow
     */
    ae_swap_matrices(a, &olda);
    ae_matrix_set_length(a, n, ae_maxint(m, mincols, _state), _state);
    for(i=0; i<=n2-1; i++)
    {
        for(j=0; j<=m-1; j++)
        {
            a->ptr.pp_double[i][j] = olda.ptr.pp_double[i][j];
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Grows X, i.e. changes its size in such a way that:
a) contents is preserved
b) new size is at least N
c) new size can be larger than N, so subsequent grow() calls can return
   without reallocation

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rvectorgrowto(/* Real    */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_vector oldx;
    ae_int_t i;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_vector_init(&oldx, 0, DT_REAL, _state, ae_true);

    
    /*
     * Enough place
     */
    if( x->cnt>=n )
    {
        ae_frame_leave(_state);
        return;
    }
    
    /*
     * Choose new size
     */
    n = ae_maxint(n, ae_round(1.8*x->cnt+1, _state), _state);
    
    /*
     * Grow
     */
    n2 = x->cnt;
    ae_swap_vectors(x, &oldx);
    ae_vector_set_length(x, n, _state);
    for(i=0; i<=n-1; i++)
    {
        if( i<n2 )
        {
            x->ptr.p_double[i] = oldx.ptr.p_double[i];
        }
        else
        {
            x->ptr.p_double[i] = (double)(0);
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Resizes X and:
* preserves old contents of X
* fills new elements by zeros

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void ivectorresize(/* Integer */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_vector oldx;
    ae_int_t i;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_vector_init(&oldx, 0, DT_INT, _state, ae_true);

    n2 = x->cnt;
    ae_swap_vectors(x, &oldx);
    ae_vector_set_length(x, n, _state);
    for(i=0; i<=n-1; i++)
    {
        if( i<n2 )
        {
            x->ptr.p_int[i] = oldx.ptr.p_int[i];
        }
        else
        {
            x->ptr.p_int[i] = 0;
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Resizes X and:
* preserves old contents of X
* fills new elements by zeros

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rvectorresize(/* Real    */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_vector oldx;
    ae_int_t i;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_vector_init(&oldx, 0, DT_REAL, _state, ae_true);

    n2 = x->cnt;
    ae_swap_vectors(x, &oldx);
    ae_vector_set_length(x, n, _state);
    for(i=0; i<=n-1; i++)
    {
        if( i<n2 )
        {
            x->ptr.p_double[i] = oldx.ptr.p_double[i];
        }
        else
        {
            x->ptr.p_double[i] = (double)(0);
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Resizes X and:
* preserves old contents of X
* fills new elements by zeros

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void rmatrixresize(/* Real    */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_matrix oldx;
    ae_int_t i;
    ae_int_t j;
    ae_int_t m2;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_matrix_init(&oldx, 0, 0, DT_REAL, _state, ae_true);

    m2 = x->rows;
    n2 = x->cols;
    ae_swap_matrices(x, &oldx);
    ae_matrix_set_length(x, m, n, _state);
    for(i=0; i<=m-1; i++)
    {
        for(j=0; j<=n-1; j++)
        {
            if( i<m2&&j<n2 )
            {
                x->ptr.pp_double[i][j] = oldx.ptr.pp_double[i][j];
            }
            else
            {
                x->ptr.pp_double[i][j] = 0.0;
            }
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
Resizes X and:
* preserves old contents of X
* fills new elements by zeros

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void imatrixresize(/* Integer */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_matrix oldx;
    ae_int_t i;
    ae_int_t j;
    ae_int_t m2;
    ae_int_t n2;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_matrix_init(&oldx, 0, 0, DT_INT, _state, ae_true);

    m2 = x->rows;
    n2 = x->cols;
    ae_swap_matrices(x, &oldx);
    ae_matrix_set_length(x, m, n, _state);
    for(i=0; i<=m-1; i++)
    {
        for(j=0; j<=n-1; j++)
        {
            if( i<m2&&j<n2 )
            {
                x->ptr.pp_int[i][j] = oldx.ptr.pp_int[i][j];
            }
            else
            {
                x->ptr.pp_int[i][j] = 0;
            }
        }
    }
    ae_frame_leave(_state);
}


/*************************************************************************
appends element to X

  -- ALGLIB --
     Copyright 20.03.2009 by Bochkanov Sergey
*************************************************************************/
void ivectorappend(/* Integer */ ae_vector* x,
     ae_int_t v,
     ae_state *_state)
{
    ae_frame _frame_block;
    ae_vector oldx;
    ae_int_t i;
    ae_int_t n;

    ae_frame_make(_state, &_frame_block);
    memset(&oldx, 0, sizeof(oldx));
    ae_vector_init(&oldx, 0, DT_INT, _state, ae_true);

    n = x->cnt;
    ae_swap_vectors(x, &oldx);
    ae_vector_set_length(x, n+1, _state);
    for(i=0; i<=n-1; i++)
    {
        x->ptr.p_int[i] = oldx.ptr.p_int[i];
    }
    x->ptr.p_int[n] = v;
    ae_frame_leave(_state);
}


/*************************************************************************
This function checks that length(X) is at least N and first N values  from
X[] are finite

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool isfinitevector(/* Real    */ ae_vector* x,
     ae_int_t n,
     ae_state *_state)
{
    ae_int_t i;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteVector: internal error (N<0)", _state);
    if( n==0 )
    {
        result = ae_true;
        return result;
    }
    if( x->cnt<n )
    {
        result = ae_false;
        return result;
    }
    for(i=0; i<=n-1; i++)
    {
        if( !ae_isfinite(x->ptr.p_double[i], _state) )
        {
            result = ae_false;
            return result;
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that first N values from X[] are finite

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool isfinitecvector(/* Complex */ ae_vector* z,
     ae_int_t n,
     ae_state *_state)
{
    ae_int_t i;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteCVector: internal error (N<0)", _state);
    for(i=0; i<=n-1; i++)
    {
        if( !ae_isfinite(z->ptr.p_complex[i].x, _state)||!ae_isfinite(z->ptr.p_complex[i].y, _state) )
        {
            result = ae_false;
            return result;
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that size of X is at least MxN and values from
X[0..M-1,0..N-1] are finite.

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool apservisfinitematrix(/* Real    */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{
    ae_int_t i;
    ae_int_t j;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteMatrix: internal error (N<0)", _state);
    ae_assert(m>=0, "APSERVIsFiniteMatrix: internal error (M<0)", _state);
    if( m==0||n==0 )
    {
        result = ae_true;
        return result;
    }
    if( x->rows<m||x->cols<n )
    {
        result = ae_false;
        return result;
    }
    for(i=0; i<=m-1; i++)
    {
        for(j=0; j<=n-1; j++)
        {
            if( !ae_isfinite(x->ptr.pp_double[i][j], _state) )
            {
                result = ae_false;
                return result;
            }
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that all values from X[0..M-1,0..N-1] are finite

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool apservisfinitecmatrix(/* Complex */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{
    ae_int_t i;
    ae_int_t j;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteCMatrix: internal error (N<0)", _state);
    ae_assert(m>=0, "APSERVIsFiniteCMatrix: internal error (M<0)", _state);
    for(i=0; i<=m-1; i++)
    {
        for(j=0; j<=n-1; j++)
        {
            if( !ae_isfinite(x->ptr.pp_complex[i][j].x, _state)||!ae_isfinite(x->ptr.pp_complex[i][j].y, _state) )
            {
                result = ae_false;
                return result;
            }
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that size of X is at least NxN and all values from
upper/lower triangle of X[0..N-1,0..N-1] are finite

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool isfinitertrmatrix(/* Real    */ ae_matrix* x,
     ae_int_t n,
     ae_bool isupper,
     ae_state *_state)
{
    ae_int_t i;
    ae_int_t j1;
    ae_int_t j2;
    ae_int_t j;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteRTRMatrix: internal error (N<0)", _state);
    if( n==0 )
    {
        result = ae_true;
        return result;
    }
    if( x->rows<n||x->cols<n )
    {
        result = ae_false;
        return result;
    }
    for(i=0; i<=n-1; i++)
    {
        if( isupper )
        {
            j1 = i;
            j2 = n-1;
        }
        else
        {
            j1 = 0;
            j2 = i;
        }
        for(j=j1; j<=j2; j++)
        {
            if( !ae_isfinite(x->ptr.pp_double[i][j], _state) )
            {
                result = ae_false;
                return result;
            }
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that all values from upper/lower triangle of
X[0..N-1,0..N-1] are finite

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool apservisfinitectrmatrix(/* Complex */ ae_matrix* x,
     ae_int_t n,
     ae_bool isupper,
     ae_state *_state)
{
    ae_int_t i;
    ae_int_t j1;
    ae_int_t j2;
    ae_int_t j;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteCTRMatrix: internal error (N<0)", _state);
    for(i=0; i<=n-1; i++)
    {
        if( isupper )
        {
            j1 = i;
            j2 = n-1;
        }
        else
        {
            j1 = 0;
            j2 = i;
        }
        for(j=j1; j<=j2; j++)
        {
            if( !ae_isfinite(x->ptr.pp_complex[i][j].x, _state)||!ae_isfinite(x->ptr.pp_complex[i][j].y, _state) )
            {
                result = ae_false;
                return result;
            }
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
This function checks that all values from X[0..M-1,0..N-1] are  finite  or
NaN's.

  -- ALGLIB --
     Copyright 18.06.2010 by Bochkanov Sergey
*************************************************************************/
ae_bool apservisfiniteornanmatrix(/* Real    */ ae_matrix* x,
     ae_int_t m,
     ae_int_t n,
     ae_state *_state)
{
    ae_int_t i;
    ae_int_t j;
    ae_bool result;


    ae_assert(n>=0, "APSERVIsFiniteOrNaNMatrix: internal error (N<0)", _state);
    ae_assert(m>=0, "APSERVIsFiniteOrNaNMatrix: internal error (M<0)", _state);
    for(i=0; i<=m-1; i++)
    {
        for(j=0; j<=n-1; j++)
        {
            if( !(ae_isfinite(x->ptr.pp_double[i][j], _state)||ae_isnan(x->ptr.pp_double[i][j], _state)) )
            {
                result = ae_false;
                return result;
            }
        }
    }
    result = ae_true;
    return result;
}


/*************************************************************************
Safe sqrt(x^2+y^2)

  -- ALGLIB --
     Copyright by Bochkanov Sergey
*************************************************************************/
double safepythag2(double x, double y, ae_state *_state)
{
    double w;
    double xabs;
    double yabs;
    double z;
    double result;


    xabs = ae_fabs(x, _state);
    yabs = ae_fabs(y, _state);
    w = ae_maxreal(xabs, yabs, _state);
    z = ae_minreal(xabs, yabs, _state);
    if( ae_fp_eq(z,(double)(0)) )
    {
        result = w;
    }
    else
    {
        result = w*ae_sqrt(1+ae_sqr(z/w, _state), _state);
    }
    return result;
}


/*************************************************************************
Safe sqrt(x^2+y^2)

  -- ALGLIB --
     Copyright by Bochkanov Sergey
*************************************************************************/
double safepythag3(double x, double y, double z, ae_state *_state)
{
    double w;
    double result;


    w = ae_maxreal(ae_fabs(x, _state), ae_maxreal(ae_fabs(y, _state), ae_fabs(z, _state), _state), _state);
    if( ae_fp_eq(w,(double)(0)) )
    {
        result = (double)(0);
        return result;
    }
    x = x/w;
    y = y/w;
    z = z/w;
    result = w*ae_sqrt(ae_sqr(x, _state)+ae_sqr(y, _state)+ae_sqr(z, _state), _state);
    return result;
}


/*************************************************************************
Safe division.

This function attempts to calculate R=X/Y without overflow.

It returns:
* +1, if abs(X/Y)>=MaxRealNumber or undefined - overflow-like situation
      (no overlfow is generated, R is either NAN, PosINF, NegINF)
*  0, if MinRealNumber<abs(X/Y)<MaxRealNumber or X=0, Y<>0
      (R contains result, may be zero)
* -1, if 0<abs(X/Y)<MinRealNumber - underflow-like situation
      (R contains zero; it corresponds to underflow)

No overflow is generated in any case.

  -- ALGLIB --
     Copyright by Bochkanov Sergey
*************************************************************************/
ae_int_t saferdiv(double x, double y, double* r, ae_state *_state)
{
    ae_int_t result;

    *r = 0;

    
    /*
     * Two special cases:
     * * Y=0
     * * X=0 and Y<>0
     */
    if( ae_fp_eq(y,(double)(0)) )
    {
        result = 1;
        if( ae_fp_eq(x,(double)(0)) )
        {
            *r = _state->v_nan;
        }
        if( ae_fp_greater(x,(double)(0)) )
        {
            *r = _state->v_posinf;
        }
        if( ae_fp_less(x,(double)(0)) )
        {
            *r = _state->v_neginf;
        }
        return result;
    }
    if( ae_fp_eq(x,(double)(0)) )
    {
        *r = (double)(0);
        result = 0;
        return result;
    }
    
    /*
     * make Y>0
     */
    if( ae_fp_less(y,(double)(0)) )
    {
        x = -x;
        y = -y;
    }
    
    /*
     *
     */
    if( ae_fp_greater_eq(y,(double)(1)) )
    {
        *r = x/y;
        if( ae_fp_less_eq(ae_fabs(*r, _state),ae_minrealnumber) )
        {
            result = -1;
            *r = (double)(0);
        }
        else
        {
            result = 0;
        }
    }
    else
    {
        if( ae_fp_greater_eq(ae_fabs(x, _state),ae_maxrealnumber*y) )
        {
            if( ae_fp_greater(x,(double)(0)) )
            {
                *r = _state->v_posinf;
            }
            else
            {
                *r = _state->v_neginf;
            }
            result = 1;
        }
        else
        {
            *r = x/y;
            result = 0;
        }
    }
    return result;
}


/*************************************************************************
This function calculates "safe" min(X/Y,V) for positive finite X, Y, V.
No overflow is generated in any case.

  -- ALGLIB --
     Copyright by Bochkanov Sergey
*************************************************************************/
double safeminposrv(double x, double y, double v, ae_state *_state)
{
    double r;
    double result;


    if( ae_fp_greater_eq(y,(double)(1)) )
    {
        
        /*
         * Y>=1, we can safely divide by Y
         */
        r = x/y;
        result = v;
        if( ae_fp_greater(v,r) )
        {
            result = r;
        }
        else
        {
            result = v;
        }
    }
    else
    {
        
        /*
         * Y<1, we can safely multiply by Y
         */
        if( ae_fp_less(x,v*y) )
        {
            result = x/y;
        }
        else
        {
            result = v;
        }
    }
    return result;
}


/*************************************************************************
This function makes periodic mapping of X to [A,B].

It accepts X, A, B (A>B). It returns T which lies in  [A,B] and integer K,
such that X = T + K*(B-A).

NOTES:
* K is represented as real value, although actually it is integer
* T is guaranteed to be in [A,B]
* T replaces X

  -- ALGLIB --
     Copyright by Bochkanov Sergey
*************************************************************************/
void apperiodicmap(double* x,
     double a,
     double b,
     double* k,
     ae_state *_state)
{

    *k = 0;

    ae_assert(ae_fp_less(a,b), "APPeriodicMap: internal error!", _state);
    *k = (double)(ae_ifloor((*x-a)/(b-a), _state));
    *x = *x-*k*(b-a);
    while(ae_fp_less(*x,a))
    {
        *x = *x+(b-a);
        *k = *k-1;
    }
    while(ae_fp_greater(*x,b))
    {
        *x = *x-(b-a);
        *k = *k+1;
    }
    *x = ae_maxreal(*x, a, _state);
    *x = ae_minreal(*x, b, _state);
}


/******************************************************************