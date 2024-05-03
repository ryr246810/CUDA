#include <fstream>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <MatDoub.hxx>

MatDoub::MatDoub() : nn(0), mm(0), v(NULL) {}

MatDoub::MatDoub(int n, int m) : nn(n), mm(m), v(n>0 ? new double*[n] : NULL)
{
  int i,nel=m*n;
  if (v) v[0] = nel>0 ? new double[nel] : NULL;
  for (i=1;i<n;i++) v[i] = v[i-1] + m;
}


MatDoub::MatDoub(int n, int m, const double &a) : nn(n), mm(m), v(n>0 ? new double*[n] : NULL)
{
  int i,j,nel=m*n;
  if (v) v[0] = nel>0 ? new double[nel] : NULL;
  for (i=1; i< n; i++) v[i] = v[i-1] + m;
  for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = a;
}


MatDoub::MatDoub(int n, int m, const double *a) : nn(n), mm(m), v(n>0 ? new double*[n] : NULL)
{
  int i,j,nel=m*n;
  if (v) v[0] = nel>0 ? new double[nel] : NULL;
  for (i=1; i< n; i++) v[i] = v[i-1] + m;
  for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = *a++;
}


MatDoub::MatDoub(const MatDoub &rhs) : nn(rhs.nn), mm(rhs.mm), v(nn>0 ? new double*[nn] : NULL)
{
  int i,j,nel=mm*nn;
  if (v) v[0] = nel>0 ? new double[nel] : NULL;
  for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
}

 
MatDoub & MatDoub::operator=(const MatDoub &rhs)
  // postcondition: normal assignment via copying has been performed;
  //		if matrix and rhs were different sizes, matrix
  //		has been resized to match the size of rhs
{
  if (this != &rhs) {
    int i,j,nel;
    if (nn != rhs.nn || mm != rhs.mm) {
      if (v != NULL) {
	delete[] (v[0]);
	delete[] (v);
      }
      nn=rhs.nn;
      mm=rhs.mm;
      v = nn>0 ? new double*[nn] : NULL;
      nel = mm*nn;
      if (v) v[0] = nel>0 ? new double[nel] : NULL;
      for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
    }
    for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
  }
  return *this;
}


void MatDoub::resize(int newn, int newm)
{
  int i,nel;
  if (newn != nn || newm != mm) {
    if (v != NULL) {
      delete[] (v[0]);
      delete[] (v);
    }
    nn = newn;
    mm = newm;
    v = nn>0 ? new double*[nn] : NULL;
    nel = mm*nn;
    if (v) v[0] = nel>0 ? new double[nel] : NULL;
    for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  }
}


void MatDoub::assign(int newn, int newm, const double& a)
{
  int i,j,nel;
  if (newn != nn || newm != mm) {
    if (v != NULL) {
      delete[] (v[0]);
      delete[] (v);
    }
    nn = newn;
    mm = newm;
    v = nn>0 ? new double*[nn] : NULL;
    nel = mm*nn;
    if (v) v[0] = nel>0 ? new double[nel] : NULL;
    for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  }
  for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = a;
}


MatDoub::~MatDoub()
{
  if (v != NULL) {
    delete[] (v[0]);
    delete[] (v);
  }
}
