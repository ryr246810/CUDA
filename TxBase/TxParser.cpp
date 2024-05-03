
#include <TxParser.h>

#include <TxStreams.h>
#include <math.h>
#include <stdlib.h>

enum SymbolKind {kindUNDEF, kindALPHA, kindNUMBER, kindPUNCT, 
                 kindOPERATOR, kindVARIABLE, kindEXPRESSION};



//***********************************************************************
// class lex
//***********************************************************************
void TxLex::removeWhiteSpaces(std::string& a)
{ 
  size_t pos = 0;
  // at present, we only consider regular spaces
  while((pos = a.find(" ", pos)) != std::string::npos) 
    a.erase(pos,1);
}

void TxLex::advanceAlpha(std::string a, size_t& pos)
{
  while(pos < a.length() && isalnum(a[pos])) 
    ++pos;
}

void TxLex::advanceInt(std::string a, size_t& pos)
{
  while(pos < a.length() && isdigit(a[pos]))
    ++pos;
}

void TxLex::advanceNumber(std::string a, size_t& pos)
{
  if(a[pos] != '.') {
    advanceInt(a, pos);
    if (pos >= a.length()) return;
  }
  
  if(a[pos] == '.') {  
    ++pos;
    advanceInt(a, pos);
    if (pos >= a.length()) return;
  }
  
  if(a[pos] == 'e' || a[pos] == 'E') {
    ++pos;
    if (pos >= a.length()) return;
    if(a[pos] =='+')  // allow signed exponent 
      ++pos; 
    else if(a[pos] == '-') 
      ++pos;
    advanceInt(a, pos);
  }
}

void TxLex::advancePunct(std::string a, size_t& pos)
{
// up to now we only support single character punctuations
  ++pos; 
}

void TxLex::getNextToken(std::string a, size_t& pos, std::string& str, int& kind)
{
  size_t posOld = pos; 

  if(isalpha(a[pos])){
    advanceAlpha(a, pos); 
    kind = kindALPHA;
  }else if(isdigit(a[pos]) || a[pos] == '.') {
    advanceNumber(a, pos); 
    kind = kindNUMBER;
  }else{
    advancePunct(a, pos);
    kind = kindPUNCT;
  }
  str = a.substr(posOld, pos-posOld);
}

//*******************************************************************
// end class lex
//*******************************************************************


// **********************************************************************
// class TxExprNode 
// **********************************************************************
 
TxExprNode::TxExprNode()
{
  symbol = "";
  value = 0.;
  kind = kindUNDEF;
  left = NULL;
  right = NULL;
}

 
TxExprNode::TxExprNode(std::string a, int k)
{
  value = 0.;
  symbol = a;
  kind = k;
  left = NULL;
  right= NULL;
}

 
double TxExprNode::stringToValue()
{
  double value = 0.;
  switch(kind) {
  case kindNUMBER : value = atof(symbol.c_str()); break; 
  case kindALPHA  : value = -1; break; 
  }

  return value;
}

 
double  TxExprNode::evaluate()
{
  double valueLeft = 0;
  double valueRight = 0;

  if(kind == kindNUMBER) return value;
  if(kind == kindVARIABLE) return *valuePtr;

  if(left != NULL) 
    valueLeft = left->evaluate(); 
  if(right != NULL)
    valueRight = right->evaluate();	   
  double val =  binOp->evaluate(valueLeft, valueRight);

  return val;
}

 
void TxExprNode::printEvalTree()
{
  std::cerr<<symbol << "(" ;
  if(left != NULL) left->printEvalTree();
  std::cerr << "|";
  if(right!= NULL) right->printEvalTree();
  std::cerr << ")";
}

 
bool TxExprNode::isConst()
{
  if(kind == kindVARIABLE) return false;
  bool isConstLeft = true;
  bool isConstRight = true;
 
  if(left != NULL) 
    isConstLeft = left->isConst();
  if(right !=NULL)
    isConstRight = right->isConst();
  return (isConstLeft & isConstRight);
}

 
void TxExprNode::prune()
{
  if(left != NULL)
    left->prune();
  if(right != NULL)
    right->prune();

  if(!isConst()) return; 
  if(left == NULL && right == NULL) return; 

  value = evaluate();

  kind = kindNUMBER;
  left = NULL;
  right = NULL;
}
//***********************************************************************
// end TxExprNode class
//***********************************************************************



//*********************************************************************
// class TxSymbolTable
//*********************************************************************
 
void TxSymbolTable::addOperator(std::string a, TxBinOp * op)
{
  TxSymbolEntry  s;
  s.kind = kindOPERATOR;
  s.binOp = op;
  symbols[a] = s;
}

 
void TxSymbolTable::addExpression(std::string a, TxExprNode * expr)
{
  TxSymbolEntry  s;
  s.kind = kindEXPRESSION;
  s.expr = expr;
  symbols[a] = s;
}

 
void TxSymbolTable::addVariable(std::string a, double* v)
{
  TxSymbolEntry  s;
  s.kind = kindVARIABLE;
  s.varPtr = v;
  symbols[a] = s;
}

 
TxBinOp* TxSymbolTable::getOperatorRef(std::string a)
{
  std::map<std::string, TxSymbolEntry >::iterator p = symbols.find(a);
  if(p != symbols.end() && p->second.kind == kindOPERATOR){ 
    return p->second.binOp; 
  }
  return NULL;
}


 
TxExprNode* TxSymbolTable::getExpressionRef(std::string a)
{
  std::map<std::string, TxSymbolEntry  >::iterator p = symbols.find(a);
  if(p != symbols.end() && p->second.kind == kindEXPRESSION){ 
    return p->second.expr; 
  }
  return NULL;
}



 
double* TxSymbolTable::getVariableRef(std::string a)
{
  std::map<std::string, TxSymbolEntry  >::iterator p = symbols.find(a);
  if(p != symbols.end() && p->second.kind == kindVARIABLE){ 
    return p->second.varPtr;
  }
  return NULL;
}
//**********************************************************************
// end class TxSymbolTable
//**********************************************************************


//*******************************************************************
//  class TxParser
//*******************************************************************
 
TxParser::TxParser()
{
  expRoot = NULL;
}

 
TxParser::TxParser(std::string a)
{
 setExpression(a);
}

 
TxParser::~TxParser()
{
  for(i=l.begin(); i!= l.end(); i++)
    delete *i;
}

 
void TxParser::setExpression(std::string a)
{
#undef PRINTTREE
#ifdef PRINTTREE
  std::cerr << " Building the tree " << std::endl;
#endif
  buildTokenList(a);
  buildEvalTree();  
  //  printTokenList();  
  buildDefSymbTable();
  buildSymbTable();
  pruneTree();
#ifdef PRINTTREE
  expRoot->printEvalTree();
  std::cerr << " finished building the tree " << std::endl;
#endif
}

 

void TxParser::buildTokenList(std::string a)
{
 size_t pos = 0;
 TxExprNode  *node;
 TxLex lex;
 std::string token;
 int kind;

 lex.removeWhiteSpaces(a);
 while(pos < a.length()){
   lex.getNextToken(a, pos, token, kind);
   node = new TxExprNode (token, kind);
   l.push_back(node);
 }
}

 
TxExprNode * TxParser::parseExpr()
{
  TxExprNode * root = parseTerm();  // parse the first expression 
  while(i != l.end() && ((*i)->symbol == "+" || (*i)->symbol == "-")){
    (*i)->left = root;                       // set first expression to LHS
    root = *i;                               // set root to operator token
    ++i;                                     // move on in token list
    root->right = parseTerm();               // set next expression to RHS
  } 
  return root;
}

 
TxExprNode * TxParser::parseTerm()
{
 TxExprNode * root = parseFact();      // parse first factor
 while(i != l.end() && ((*i)->symbol == "*" || (*i)->symbol == "/")){
   (*i)->left = root;                 // set first factor to LHS
   root = *i;                         // set root to operator token
   ++i;                               // move to next token in list
   root->right = parseFact();         // set the next factor to RHS
 } 
 return root;
}

 
TxExprNode * TxParser::parseFact()
{
  TxExprNode * root = parseSymb();
  if(i == l.end()) return root;
  while(i!=l.end() && ((*i)->symbol == "^")){
    (*i)->left = root;
    root = *i;
    ++i;
    if(i == l.end()) std::cout << "PARSE ERROR: Missing exponent!" << std::endl; 
    root->right = parseSymb();
  }
  return root;
}

 
TxExprNode * TxParser::parseSymb()
{
 TxExprNode * root;

 if((*i)->symbol == "("){
   ++i;	 
   root = parseExpr();
   if(i == l.end() || (*i)->symbol != ")") 
	   std::cout << "PARSE ERROR: Parenthesis imbalance!" << std::endl;
   ++i;
   return root;
 }  

 if((*i)->symbol == "-") {
   (*i)->symbol = "inv";
   root = *i;
   ++i;
   TxExprNode * le = parseFact();
   root->left = le;
   root->right = NULL;
   return root;
 }

 if((*i)->symbol == "+"){
   ++i;
   root = parseFact();
   return root;
 }

 if((*i)->kind == kindALPHA){
   root = *i;
   ++i;
   if(i == l.end()) return root;
   if((*i)->symbol == "("){
     ++i;
     TxExprNode * le = parseExpr();
     if((*i)->symbol != ")") 
       std::cout << "PARSE ERROR: function argument" << std::endl;
     ++i;
     root->left = le;
     root->right = NULL;
   }
   return root;
 } 

 if((*i)->kind == kindNUMBER){
  root = *i; 
  ++i;
  return root;
 } 

 return NULL;
}

 
void TxParser::buildEvalTree() 
{
 i = l.begin();
 expRoot = parseExpr();
}

 
void TxParser::printTokenList()
{
 for(i = l.begin(); i != l.end(); i++)
   std::cout << (**i).symbol << "  node=" << (*i) << "  left=" 
	     << (**i).left << " right=" << (**i).right <<std::endl;
}

 
void TxParser::assignValue(std::string a, double val)
{
  double* varPtr = symbolTable.getVariableRef(a);
  if(varPtr != NULL){
    *varPtr = val;
  } else {
    std::cout << "assign Value: Variable " << a << " not found!" << std::endl;
  } 
}

 
double* TxParser::getValuePtr(std::string a)
{
 return  symbolTable.getVariableRef(a);
}

 
double TxParser::evaluate() const
{
  return expRoot->evaluate();
}

 
void TxParser::pruneTree()
{
  expRoot->prune();
}


 
void TxParser::buildDefSymbTable()
{
  symbolTable.addOperator("+", &opPlus);
  symbolTable.addOperator("-", &opMinus);
  symbolTable.addOperator("*", &opTimes);
  symbolTable.addOperator("/", &opDiv);
  symbolTable.addOperator("^", &opPow);
  symbolTable.addOperator("sin", &opSin);
  symbolTable.addOperator("cos", &opCos);
  symbolTable.addOperator("exp", &opExp);
  symbolTable.addOperator("tan", &opTan);
  symbolTable.addOperator("asin", &opAsin);
  symbolTable.addOperator("acos", &opAcos);
  symbolTable.addOperator("ln", &opLn);
  symbolTable.addOperator("atan", &opAtan);
  symbolTable.addOperator("inv", &opInv);
  symbolTable.addOperator("H",   &opH);
  symbolTable.addOperator("abs", &opAbs);
  symbolTable.addOperator("sqrt", &opSqrt);
  symbolTable.addOperator("j0", &opj0);
  symbolTable.addOperator("j1", &opj1);

}

 
void TxParser::buildSymbTable()
{
  for(i=l.begin(); i!=l.end(); i++){
    if((*i)->kind == kindNUMBER) {
      (*i)->value = (*i)->stringToValue();
      continue;
    }


    TxExprNode * en = symbolTable.getExpressionRef((*i)->symbol);
    if(en != NULL) {
      (*i)->kind = kindEXPRESSION;
      (*i)->left = en;
      (*i)->right = NULL;
    }

    TxBinOp * bo = symbolTable.getOperatorRef((*i)->symbol);
    if(bo != NULL){
      (*i)->binOp = bo;
      (*i)->kind = kindOPERATOR; 
      continue;
    }
    
    (*i)->kind = kindVARIABLE;
    double* vp = symbolTable.getVariableRef((*i)->symbol);
    if(vp != NULL){
      (*i)->valuePtr = vp;
    } else {
      double* vPtr = &((*i)->value);
      symbolTable.addVariable((*i)->symbol, vPtr);
      (*i)->valuePtr = vPtr;
    }
  }   
}

//*************************************************************
// end class parser
//*************************************************************
