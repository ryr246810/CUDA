
#ifndef _VecDoub_HeaderFile
#define _VecDoub_HeaderFile

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



class VecDoub 
{
private:
  int nn;	// size of array. upper index is nn-1
  double *v;
public:
  VecDoub();
  explicit VecDoub(int n);		// Zero-based array
  VecDoub(int n, const double &a);	//initialize to constant value
  VecDoub(int n, const double *a);	// Initialize to array
  VecDoub(const VecDoub &rhs);	// Copy constructor
  VecDoub & operator=(const VecDoub &rhs);	//assignment
  typedef double value_type; // make double available externally
  inline double & operator[](const int i);	//i'th element
  inline const double & operator[](const int i) const;
  inline int size() const;
  void resize(int newn); // resize (contents not preserved)
  void assign(int newn, const double &a); // resize and assign a constant value
  ~VecDoub();
};



inline double & VecDoub::operator[](const int i)	//subscripting
{
#ifdef _CHECKBOUNDS_
  if (i<0 || i>=nn) {
    throw("VecDoub subscript out of bounds");
  }
#endif
  return v[i];
}


inline const double & VecDoub::operator[](const int i) const	//subscripting
{
#ifdef _CHECKBOUNDS_
  if (i<0 || i>=nn) {
    throw("VecDoub subscript out of bounds");
  }
#endif
  return v[i];
}


inline int VecDoub::size() const
{
  return nn;
}


// end of VecDoub definitions

#endif
