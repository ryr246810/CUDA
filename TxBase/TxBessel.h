#ifndef _TXBESSEL_H_
#define _TXBESSEL_H_


#include <cmath>
#include <limits>

namespace Bess{
  double bess_j0(const double x);
  double bess_j1(const double x);
  double bess_y0(const double x);
  double bess_y1(const double x);
  
  void bess_rat(const double x, const double *r, const double *s, const int n,
		double& y, double& z, double& nump, double& denp);
  
  void bess_asp(const double *pn, const double *pd, const double *qn, const double *qd, const double fac,
		const double ax,
		double& y, double& z, double& xx,  double& nump, double& denp, double& numq, double& denq);

  double bess_i0(const double x);
  
  double bess_i1(const double x);
  
  double bess_k0(const double x);
  
  double bess_k1(const double x);
  
  double bess_poly(const double *cof, const int n, const double x);
};

#endif

