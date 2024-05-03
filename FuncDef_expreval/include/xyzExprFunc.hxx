#ifndef _xyzExprFunc_HeaderFile
#define _xyzExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>

using namespace std;
using namespace ExprEval;


class xyzExprFunc : public ExprFuncBase
{
public:
  xyzExprFunc();
  ~xyzExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute(const double x, const double y, const double z);

protected:
  double* m_x;
  double* m_y;
  double* m_z;
};

#endif
