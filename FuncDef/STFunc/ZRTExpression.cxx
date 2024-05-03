
#include <ZRTExpression.hxx>

ZRTExpression::
ZRTExpression()
{
  expression = "0.";
  dummyVar = 0.;
}

void 
ZRTExpression::
setAttrib(const TxHierAttribSet& tas)
{
  // Get the starting and ending positions
  if(tas.hasString("expression")) 
    expression = tas.getString("expression");
  
  parser = new TxParser(expression);
  xPtr[0] = parser->getValuePtr("z");
  xPtr[1] = parser->getValuePtr("r");
  
  for(size_t i=0; i<2; ++i) 
    if(xPtr[i] == NULL) xPtr[i] = &dummyVar;
  
  tPtr = parser->getValuePtr("t");
  if(tPtr == NULL) tPtr = &dummyVar;
}


Standard_Real 
ZRTExpression::
operator()(Standard_Real* x, Standard_Real t) const
{
  // copy the arguments into the parse tree (or into dummyVar, if not used in parse Tree
  for(size_t i = 0; i < 2; i++)
    *(xPtr[i]) = x[i];
  *tPtr = t;
  
  Standard_Real res = parser->evaluate();
  return res;
}

