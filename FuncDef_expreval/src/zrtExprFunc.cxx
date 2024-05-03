#include <zrtExprFunc.hxx>

zrtExprFunc::zrtExprFunc()
  : ExprFuncBase()
{
  m_z = NULL;
  m_r = NULL;
  m_t = NULL;
}

zrtExprFunc::
~zrtExprFunc()
{
  if(m_z!=NULL) delete m_z;
  if(m_r!=NULL) delete m_r;
  if(m_t!=NULL) delete m_t;
}

void
zrtExprFunc::
init()
{
  m_z = new double;
  m_r = new double;
  m_t = new double;

  m_vlist.AddAddress("z", m_z);
  m_vlist.AddAddress("r", m_r);
  m_vlist.AddAddress("t", m_t);

  ExprFuncBase::init();  
}


double
zrtExprFunc::
evalute(const double z, const double r, const double t)
{
  *m_z = z;
  *m_r = r;
  *m_t = t;
  double result = m_expression.Evaluate();

  return result;
}


void
zrtExprFunc::
setAttrib(const TxHierAttribSet& tas)
{
  ExprFuncBase::setAttrib(tas);
  m_expression.Parse(m_InputString);
  this->evalute(1.0, 1.0, 1.0);
}
