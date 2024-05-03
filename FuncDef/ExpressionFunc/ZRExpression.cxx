
#include <ZRExpression.hxx>

ZRExpression::
ZRExpression()
  : ExpressionFuncDef()
{

}



ZRExpression::
~ZRExpression()
{

}


void 
ZRExpression::
setAttrib(const TxHierAttribSet& tas)
{
  m_params.insert( std::pair<const std::string, double>("z", 0.0) );
  m_params.insert( std::pair<const std::string, double>("r", 0.0) );

  ExpressionFuncDef::setAttrib(tas);
}


Standard_Real 
ZRExpression::
operator()(const Standard_Real z, const Standard_Real r)
{
  m_params["z"] = z;
  m_params["r"] = r;

  m_dfl->evaluate();
  Standard_Real res = (m_dfl->operator[]("result")).getValue();
  return res;
}

