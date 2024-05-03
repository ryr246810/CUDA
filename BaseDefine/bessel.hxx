#ifndef _BESSEL_H_
#define _BESSEL_H_


#include <cmath>
#include <limits>

struct Bessjy {
  static const double xj00,xj10,xj01,xj11,twoopi,pio4;
  static const double j0r[7],j0s[7],j0pn[5],j0pd[5],j0qn[5],j0qd[5];
  static const double j1r[7],j1s[7],j1pn[5],j1pd[5],j1qn[5],j1qd[5];
  static const double y0r[9],y0s[9],y0pn[5],y0pd[5],y0qn[5],y0qd[5];
  static const double y1r[8],y1s[8],y1pn[5],y1pd[5],y1qn[5],y1qd[5];
  double nump,denp,numq,denq,y,z,ax,xx;
  
  double j0(const double x);

  double j1(const double x);
  
  double y0(const double x);
  
  double y1(const double x);
  
  double jn(const int n, const double x);
  
  double yn(const int n, const double x);
  
  void rat(const double x, const double *r, const double *s, const int n);
  
  void asp(const double *pn, const double *pd, const double *qn, const double *qd, const double fac);
};


struct Bessik {
  static const double i0p[14],i0q[5],i0pp[5],i0qq[6];
  static const double i1p[14],i1q[5],i1pp[5],i1qq[6];
  static const double k0pi[5],k0qi[3],k0p[5],k0q[3],k0pp[8],k0qq[8];
  static const double k1pi[5],k1qi[3],k1p[5],k1q[3],k1pp[8],k1qq[8];
  double y,z,ax,term;
  
  double i0(const double x);
  
  double i1(const double x);
  
  double k0(const double x);
  
  double k1(const double x);
  
  double in(const int n, const double x);
  
  double kn(const int n, const double x);
  
  inline double poly(const double *cof, const int n, const double x) {
    double ans = cof[n];
    for (int i=n-1;i>=0;i--) ans = ans*x+cof[i];
    return ans;
  }
};

#endif

