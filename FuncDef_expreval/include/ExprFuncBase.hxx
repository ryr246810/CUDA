#ifndef _ExprFuncBase_HeaderFile
#define _ExprFuncBase_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>


using namespace std;
using namespace ExprEval;


class ExprFuncBase
{
public:
  ExprFuncBase();
  virtual ~ExprFuncBase();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);

  virtual double evalute();    
  virtual double evalute(const double arg1);  
  virtual double evalute(const double arg1, const double arg2);
  virtual double evalute(const double arg1, const double arg2, const double arg3);
  virtual double evalute(const double arg1, const double arg2, const double arg3, const double arg4);

protected:
  ValueList m_vlist;
  FunctionList m_flist;
  Expression m_expression;
  string m_InputString;
};

#endif
