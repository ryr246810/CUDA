#include <tExprFunc.hxx>


tExprFunc::tExprFunc()
  : ExprFuncBase()  
{
  m_t = NULL;
  m_T = NULL;
}


tExprFunc::
~tExprFunc()
{
  if(m_t!=NULL) delete m_t;
  if(m_T!=NULL) delete m_T;
}


void
tExprFunc::
init()
{
  ExprFuncBase::init();
  m_t = new double;
  m_T = new double;
  m_vlist.AddAddress("t", m_t);
  m_vlist.AddAddress("T", m_T);
}


double
tExprFunc::
evalute(const double t)
{
  *m_t = t;
  *m_T = t;
  double result = m_expression.Evaluate();
  return result;
}


void
tExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute(1.0);
}
