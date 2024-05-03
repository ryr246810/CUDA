#ifndef _noArgExprFunc_HeaderFile
#define _noArgExprFunc_HeaderFile

#include <TxHierAttribSet.h>
#include <expreval.h>

#include <ExprFuncBase.hxx>


using namespace std;
using namespace ExprEval;


class noArgExprFunc : public ExprFuncBase
{
public:
  noArgExprFunc();
  virtual ~noArgExprFunc();

  virtual void init();
  virtual void setAttrib(const TxHierAttribSet& tas);
  
  virtual double evalute( );
};

#endif
