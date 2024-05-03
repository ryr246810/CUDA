#ifndef _tExprFunc_HeaderFile
#define _tExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>


using namespace std;
using namespace ExprEval;


class tExprFunc : public ExprFuncBase
{
public:
  tExprFunc();
  ~tExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute(const double t);

protected:
  double* m_t;
  double* m_T;
};

#endif
