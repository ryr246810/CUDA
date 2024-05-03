
#include <XYZExpression.hxx>

XYZExpression::
XYZExpression()
  : ExpressionFuncDef()
{

}


XYZExpression::
~XYZExpression()
{

}


void 
XYZExpression::
setAttrib(const TxHierAttribSet& tas)
{
  m_params.insert( std::pair<const std::string, double>("x", 0.0) );
  m_params.insert( std::pair<const std::string, double>("y", 0.0) );
  m_params.insert( std::pair<const std::string, double>("z", 0.0) );

  ExpressionFuncDef::setAttrib(tas);
}


Standard_Real 
XYZExpression::
operator()(const Standard_Real x, const Standard_Real y, const Standard_Real z)
{
  m_params["x"] = x;
  m_params["y"] = y;
  m_params["z"] = z;

  m_dfl->evaluate();
  Standard_Real res = (m_dfl->operator[]("result")).getValue();
  return res;
}

