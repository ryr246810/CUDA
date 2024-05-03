#ifndef _zrtExprFunc_HeaderFile
#define _zrtExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>


using namespace std;
using namespace ExprEval;


class zrtExprFunc : public ExprFuncBase
{
public:
  zrtExprFunc();
  ~zrtExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute(const double z, const double r, const double t);

protected:
  double* m_z;
  double* m_r;
  double* m_t;
};

#endif
