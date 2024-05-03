
#include <TExpression.hxx>

TExpression::
TExpression()
{
  expression = "0.";
  dummyVar = 0.;
}

void 
TExpression::
setAttrib(const TxHierAttribSet& tas)
{
  // Get the starting and ending positions
  if(tas.hasString("expression")) 
    expression = tas.getString("expression");
  
  parser = new TxParser(expression);

  tPtr = parser->getValuePtr("t");
  if(tPtr == NULL) tPtr = &dummyVar;
  
}


Standard_Real 
TExpression::
operator()(Standard_Real t) const
{
  *tPtr = t;
  
  Standard_Real res = parser->evaluate();
  return res;
}


