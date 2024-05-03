#ifndef _MatDoub_HeaderFile
#define _MatDoub_HeaderFile

#include <fstream>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>

class MatDoub
{
private:
  int nn;
  int mm;
  double **v;
public:
  MatDoub();
  MatDoub(int n, int m);			// Zero-based array
  MatDoub(int n, int m, const double &a);	//Initialize to constant
  MatDoub(int n, int m, const double *a);	// Initialize to array
  MatDoub(const MatDoub &rhs);		// Copy constructor
  MatDoub & operator=(const MatDoub &rhs);	//assignment
  typedef double value_type; // make double available externally
  inline double* operator[](const int i);	//subscripting: pointer to row i
  inline const double* operator[](const int i) const;
  inline int nrows() const;
  inline int ncols() const;
  void resize(int newn, int newm); // resize (contents not preserved)
  void assign(int newn, int newm, const double &a); // resize and assign a constant value
  ~MatDoub();
};


inline double* MatDoub::operator[](const int i)	//subscripting: pointer to row i
{
#ifdef _CHECKBOUNDS_
  if (i<0 || i>=nn) {
    throw("MatDoub subscript out of bounds");
  }
#endif
  return v[i];
}


inline const double* MatDoub::operator[](const int i) const
{
#ifdef _CHECKBOUNDS_
  if (i<0 || i>=nn) {
    throw("MatDoub subscript out of bounds");
  }
#endif
  return v[i];
}


inline int MatDoub::nrows() const
{
  return nn;
}


inline int MatDoub::ncols() const
{
  return mm;
}


#endif
