#include <noArgExprFunc.hxx>


noArgExprFunc::noArgExprFunc()
  : ExprFuncBase()  
{

}


noArgExprFunc::
~noArgExprFunc()
{
}


void
noArgExprFunc::
init()
{
  ExprFuncBase::init();
}


double
noArgExprFunc::
evalute()
{
  double result = ExprFuncBase::evalute();
  return result;
}

void
noArgExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute();
}
