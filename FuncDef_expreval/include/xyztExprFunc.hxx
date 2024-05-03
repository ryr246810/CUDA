#ifndef _xyztExprFunc_HeaderFile
#define _xyztExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>

using namespace std;
using namespace ExprEval;


class xyztExprFunc : public ExprFuncBase
{
public:
  xyztExprFunc();
  virtual ~xyztExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute(const double x, const double y, const double z, const double t);

protected:
  double* m_x;
  double* m_y;
  double* m_z;
  double* m_t;
};

#endif
