#include <xyzExprFunc.hxx>


xyzExprFunc::xyzExprFunc()
  : ExprFuncBase()
{
  m_x = NULL;
  m_y = NULL;
  m_z = NULL;
}


xyzExprFunc::
~xyzExprFunc()
{
  if(m_x!=NULL) delete m_x;
  if(m_y!=NULL) delete m_y;
  if(m_z!=NULL) delete m_z;
}


void
xyzExprFunc::
init()
{
  m_x = new double;
  m_y = new double;
  m_z = new double;

  m_vlist.AddAddress("x", m_x);
  m_vlist.AddAddress("y", m_y);
  m_vlist.AddAddress("z", m_z);

  ExprFuncBase::init();
}


double
xyzExprFunc::
evalute(const double x, const double y, const double z)
{
  *m_x = x;
  *m_y = y;
  *m_z = z;

  double result = m_expression.Evaluate();
  return result;
}


void
xyzExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute(1.0, 1.0, 1.0);
}
