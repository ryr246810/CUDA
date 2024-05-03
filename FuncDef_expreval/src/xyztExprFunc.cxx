#include <xyztExprFunc.hxx>


xyztExprFunc::xyztExprFunc()
  : ExprFuncBase()
{
  m_x = NULL;
  m_y = NULL;
  m_z = NULL;
  m_t = NULL;
}


xyztExprFunc::
~xyztExprFunc()
{
  if(m_x!=NULL) delete m_x;
  if(m_y!=NULL) delete m_y;
  if(m_z!=NULL) delete m_z;
  if(m_t!=NULL) delete m_t;
}


void
xyztExprFunc::
init()
{
  m_x = new double;
  m_y = new double;
  m_z = new double;
  m_t = new double;

  m_vlist.AddAddress("t", m_t);
  m_vlist.AddAddress("x", m_x);
  m_vlist.AddAddress("y", m_y);
  m_vlist.AddAddress("z", m_z);

  ExprFuncBase::init();
}


double
xyztExprFunc::
evalute(const double x, const double y, const double z, const double t)
{
  *m_x = x;
  *m_y = y;
  *m_z = z;
  *m_t = t;
  double result = m_expression.Evaluate();

  return result;
}



void
xyztExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute(1.0, 1.0, 1.0, 1.0);
}
