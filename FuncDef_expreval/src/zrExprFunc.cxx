#include <zrExprFunc.hxx>

zrExprFunc::zrExprFunc()
  : ExprFuncBase()
{
  m_z = NULL;
  m_r = NULL;
  m_Z = NULL;
  m_R = NULL;
}


zrExprFunc::
~zrExprFunc()
{
  if(m_z!=NULL) delete m_z;
  if(m_r!=NULL) delete m_r;
  if(m_Z!=NULL) delete m_Z;
  if(m_R!=NULL) delete m_R;
}


void
zrExprFunc::
init()
{
  m_z = new double;
  m_r = new double;
  m_Z = new double;
  m_R = new double;

  m_vlist.AddAddress("z", m_z);
  m_vlist.AddAddress("r", m_r);

  m_vlist.AddAddress("Z", m_Z);
  m_vlist.AddAddress("R", m_R);
  ExprFuncBase::init();  
}


double
zrExprFunc::
evalute(const double z, const double r)
{
  *m_z = z;
  *m_r = r;
  *m_Z = z;
  *m_R = r;
  double result = m_expression.Evaluate();
  return result;
}


void
zrExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute(1.0, 1.0);
}
