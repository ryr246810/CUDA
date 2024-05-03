#include <BaseFunctionDefine.hxx>
#include <PhysConsts.hxx>
#include <cmath>
#include <iostream>
#include "BaseDataDefine.hxx"
#include <string.h>

Cempic_uint64 cycle_s, cycle_e;
Cempic_uint64 worktime[100] = {0};

Standard_Integer ThreeDim_DirBump(Standard_Integer dir0, Standard_Integer num)
{
  return (dir0+num)%3;
}


Standard_Integer TwoDim_DirBump(Standard_Integer dir0, Standard_Integer num)
{
  return (dir0+num)%2;
}


Standard_Integer CurrIndexOfCircularIndices(const Standard_Integer nb,
					    const Standard_Integer currIndex)
{
  Standard_Integer resultIndex;

  if(currIndex >= nb){
    resultIndex = currIndex - nb;
  }else{
    resultIndex = currIndex;
  }

  return resultIndex;;
}



Standard_Integer PrevIndexOfCircularIndices(const Standard_Integer nb,
					    const Standard_Integer currIndex)
{
  Standard_Integer prevIndex;

  if(currIndex == 0){
    prevIndex = nb-1;
  }else{
    prevIndex = currIndex-1;
  }

  return prevIndex;
}



Standard_Integer NextIndexOfCircularIndices(const Standard_Integer nb,
					    const Standard_Integer currIndex)
{
  Standard_Integer nextIndex;

  if(currIndex == (nb-1)){
    nextIndex = 0;
  }else{
    nextIndex = currIndex+1;
  }

  return nextIndex;
}


bool Bit_Set_BoolOpt_AND(const Standard_Integer theRef, const std::set<Standard_Integer>& theSet)
{
  bool result = false;

  std::set<Standard_Integer>::const_iterator iter;
  for(iter = theSet.begin(); iter!=theSet.end(); iter++){
    Standard_Integer theOne = *iter;
    if( (theRef & theOne) !=0){
      result = true;
      break;
    }
  }
  return result;
}


void v_to_u(const double v, double& u)
{
  /*
  double c = 2.99792458e8;
  double gamma = 1.0/sqrt(1-(v/c)*(v/c));
  //*/

  double gamma = 1.0/sqrt(1-v*v*mksConsts.c2inv);
  u = v*gamma;
}




void u_to_v(const double u, double& v)
{
  //double c = 2.99792458e8;
  double gamma = sqrt(1+u*u*mksConsts.c2inv);
  v = u/gamma;
}



void phi_To_v(const double q, const double m, const double phi, double& v)
{
  /*
  double c = 2.99792458e8;
  double c2 = c*c;
  double c4=c2*c2;

  double b = 2.0*fabs(phi*q/m);
  double b2 = b*b;
  v = 1.0/sqrt(2.0)*b/c*sqrt(sqrt(1.0+4.0*c4/b2)-1.0);
  //*/

  /*
  double c = 2.99792458e8;
  v = sqrt(1.0-1.0/(1.0+phi/0.511e6)/(1.0+phi/0.511e6));
  v = v*c;
  //*/

  double c = 2.99792458e8;
  double tmp = 1.0+phi/0.511e6;
  tmp *= tmp; 
  v = sqrt(1.0-1.0/tmp);
  v = v*c;
}


void phi_To_u(const double q, const double m, const double phi, double& u)
{
  double v;
  phi_To_v(q, m, phi, v);
  v_to_u(v,u);
}

double energyMKS(double u, double m)
{
  double u2 = u * u;
  double energyMKS = u2 * m * (1.0 + sqrt(1.0 + iSPEED_OF_LIGHT_SQ*u2));
  return energyMKS;
}

void* aligned_malloc(size_t required_bytes)
{
    int offset = ALIGNMENT - 1 + sizeof(void*);
    void* p1 = (void*)malloc(required_bytes + offset);
    if (p1 == NULL)
        return NULL;
    void** p2 = (void**)( ( (size_t)p1 + offset ) & ~(ALIGNMENT - 1) );
    p2[-1] = p1;
    return p2;
}
 
void aligned_free(void *p2)
{
    void* p1 = ((void**)p2)[-1];
    free(p1);
}

void malloc_Double2D_aligned(double ***head, int X, int Y){
  (*head) = (double **) (aligned_malloc(sizeof(double*) * X));
  for (int i=0; i<X;i++){
    (*head)[i] = (double *) (aligned_malloc(sizeof(double) * Y));;
  }
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = 0.0;
    }
  }
}

void malloc_Double2D_aligned_continuespace(double ***head, int X, int Y){
  (*head) = (double **) (aligned_malloc(sizeof(double*) * X));
  double * tmp1 = (double *) (aligned_malloc(sizeof(double) * X * Y));
  for (int i=0; i<X;i++){
    (*head)[i] = tmp1 + i*Y;
  }
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = 0.0;
    }
  }
  memset((*head)[0],0,sizeof(double)*X*Y);
}

void free_Double2D_aligned_continuespace(double ***head){
  aligned_free((*head)[0]);
  aligned_free((*head));
}

void malloc_Double3D_aligned_continuespace(double ****head, int X, int Y, int Z){
  (*head) = (double ***) (aligned_malloc(sizeof(double*) * X));
  double ** tmp1 = (double **) (aligned_malloc(sizeof(double*) * X * Y));
  for (int i=0; i<X;i++){
      (*head)[i] = tmp1 + i*Y;
  }
  double * tmp2 = (double *) (aligned_malloc(sizeof(double) * X * Y * Z));
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = tmp2 + (i*Y+j)*Z;
    }
  }
  memset((*head)[0][0],0,sizeof(double)*X*Y*Z);
}

void free_Double3D_aligned_continuespace(double ****head){
  aligned_free((*head)[0][0]);
  aligned_free((*head)[0]);
  aligned_free((*head));
}

void free_Double2D_aligned(double ***head, int X, int Y){
  for (int i=0; i<X;i++){
    if ((*head)[i]!=NULL) aligned_free((*head)[i]);
  }
  if ((*head)!=NULL) aligned_free((*head));
}

void malloc_Int2D_aligned(int ***head, int X, int Y){
  (*head) = (int **) (aligned_malloc(sizeof(int*) * X));
  for (int i=0; i<X;i++){
    (*head)[i] = (int *) (aligned_malloc(sizeof(int) * Y));;
  }
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = 0;
    }
  }
}

void free_Int2D_aligned(int ***head, int X, int Y){
  for (int i=0; i<X;i++){
    if ((*head)[i]!=NULL) aligned_free((*head)[i]);
  }
  if ((*head)!=NULL) aligned_free((*head));
}

void malloc_Double2D(double ***head, int X, int Y){
  (*head) = new double* [X];
  for (int i=0; i<X;i++){
    (*head)[i] = new double[Y];
  }
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = 0.0;
    }
  }
}

void free_Double2D(double ***head, int X, int Y){
  for (int i=0; i<X;i++){
    if ((*head)[i]!=NULL) delete[] (*head)[i];
  }
  if ((*head)!=NULL) delete[] (*head);
}

void malloc_Int2D(int ***head, int X, int Y){
  (*head) = new int* [X];
  for (int i=0; i<X;i++){
    (*head)[i] = new int[Y];
  }
  for (int i=0;i<X;i++)
  {
    for (int j=0;j<Y;j++){
      (*head)[i][j] = 0;
    }
  }
}

void free_Int2D(int ***head, int X, int Y){
  for (int i=0; i<X;i++){
    if ((*head)[i]!=NULL) delete[] (*head)[i];
  }
  if ((*head)!=NULL) delete[] (*head);
}
