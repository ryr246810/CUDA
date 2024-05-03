
#include <fstream>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <VecDoub.hxx>

// VecDoub definitions

VecDoub::VecDoub() : nn(0), v(NULL) {}


VecDoub::VecDoub(int n) : nn(n), v(n>0 ? new double[n] : NULL) {}


VecDoub::VecDoub(int n, const double& a) : nn(n), v(n>0 ? new double[n] : NULL)
{
  for(int i=0; i<n; i++) v[i] = a;
}


VecDoub::VecDoub(int n, const double *a) : nn(n), v(n>0 ? new double[n] : NULL)
{
  for(int i=0; i<n; i++) v[i] = *a++;
}


VecDoub::VecDoub(const VecDoub &rhs) : nn(rhs.nn), v(nn>0 ? new double[nn] : NULL)
{
  for(int i=0; i<nn; i++) v[i] = rhs[i];
}


VecDoub & VecDoub::operator=(const VecDoub &rhs)
// postcondition: normal assignment via copying has been performed;
//		if vector and rhs were different sizes, vector
//		has been resized to match the size of rhs
{
  if (this != &rhs)
    {
      if (nn != rhs.nn) {
	if (v != NULL) delete [] (v);
	nn=rhs.nn;
	v= nn>0 ? new double[nn] : NULL;
      }
      for (int i=0; i<nn; i++)
	v[i]=rhs[i];
    }
  return *this;
}



void VecDoub::resize(int newn)
{
  if (newn != nn) {
    if (v != NULL) delete[] (v);
    nn = newn;
    v = nn > 0 ? new double[nn] : NULL;
  }
}


void VecDoub::assign(int newn, const double& a)
{
  if (newn != nn) {
    if (v != NULL) delete[] (v);
    nn = newn;
    v = nn > 0 ? new double[nn] : NULL;
  }
  for (int i=0;i<nn;i++) v[i] = a;
}


VecDoub::~VecDoub()
{
  if (v != NULL) delete[] (v);
}
