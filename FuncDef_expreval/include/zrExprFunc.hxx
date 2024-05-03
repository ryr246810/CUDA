#ifndef _zrExprFunc_HeaderFile
#define _zrExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>


using namespace std;
using namespace ExprEval;


class zrExprFunc : public ExprFuncBase
{
public:
  zrExprFunc();
  ~zrExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute(const double z, const double r);

protected:
  double* m_z;
  double* m_r;
  double* m_Z;
  double* m_R;
};

#endif
