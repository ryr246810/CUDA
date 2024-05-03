// --------------------------------------------------------------------
//
// File:	TxParser.h
//
// Purpose:	Interface of an arbitrary expression parser
//
// --------------------------------------------------------------------
#ifndef _TxParser_HeaderFile
#define _TxParser_HeaderFile


#include <math.h>

#include <iostream>
#include <string>
#include <list>
#include <map>
#include <cctype>

/**
 * prototype for binary operator. Will  be used in the symbol table.
 * Avoids the use of a void* ptr.
 */

class TxBinOp
{
 public:
  virtual double evaluate(double x, double y) = 0;
};

/**
 * binary operator class, implementing addition
 */

class TxBinOpPlus:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return x + y; }  
};

/**
 * binary operator class, implementing subtraction
 */

class TxBinOpMinus:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return x - y; }  
};

/**
 * binary operator class, implementing multiplication
 */
class TxBinOpTimes:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return x * y; }  
};

/**
 * binary operator class, implementing division
 */

class TxBinOpDiv:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return x / y; }  
};

/**
 * binary operator class, implementing exponentiation
 */

class TxBinOpPow:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return pow(x, y); }  
};
 
/**
 * binary operator class, implementing a sign change
 */

class TxBinOpInv:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return -x; }  
};

/**
 * binary operator class, implementing sin
 */

class TxBinOpSin:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return sin(x); }  
};

/**
 * binary operator class, implementing cos
 */

class TxBinOpCos:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return cos(x); }  
};

/**
 * binary operator class, implementing tan
 */

class TxBinOpTan:public TxBinOp
{
 public:
  double evaluate(double x, double y){ return tan(x); }  
};

/**
 * binary operator class, implementing exp
 */

class TxBinOpExp:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return exp(x); }  
};


/**
 * binary operator class, implementing asin
 */

class TxBinOpAsin:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return asin(x); }  
};


/**
 * binary operator class, implementing acos
 */

class TxBinOpAcos:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return acos(x); }  
};

/**
 * binary operator class, implementing alog
 */
class TxBinOpLn:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return log(x); }  
};

/**
 * binary operator class, implementing atan
 */
class TxBinOpAtan:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return atan(x); }  
};

/**
 *  binary operator class, implementing heaviside 
 */
class TxBinOpH:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ return (x > 0.) ? 1. : 0.; }
};

/**
 *  binary operator class, implementing absolute value
 */

class TxBinOpAbs:public TxBinOp 
{
 public:
  double evaluate(double x, double y){ 
    return fabs(x); 
  }
};


/**
 *  binary operator class, implementing absolute value
 */

class TxBinOpSqrt:public TxBinOp
{
 public:
  double evaluate(double x, double y){ 
    return sqrt(x); 
  }
};

class TxBinOpj0:public TxBinOp
{
 public:
  double evaluate(double x, double y){ 
    return j0(x); 
  }
};

class TxBinOpj1:public TxBinOp
{
 public:
  double evaluate(double x, double y){ 
    return j1(x); 
  }
};



/**
 * class implementing a simple lexiacal analyzer
 */
class TxLex
{
 public:
  /**
   * remove all white spaces in a string. This assumes that tokens are not
   * separated by white spaces only.
   */
  void removeWhiteSpaces(std::string& a);
  
  /**
   * advances the input stream by one alpha token
   * [a-zA-Z]([a-zA-Z])*
   */
  void advanceAlpha(std::string a, size_t& pos);
  /**
   * advances the input stream by one integer token 
   * [0-9](0-9)*
   */
  void advanceInt(std::string a, size_t& pos);
  /**
   * advances the input stream by one number token
   * integer(.integer)([eE]integer)
   */
  void advanceNumber(std::string a, size_t& pos);
  /**
   * advance the input stream by a punctuation token
   * (currently just advances by a single character)
   */
  void advancePunct(std::string a, size_t& pos);
  /**
   * returns the next token and its kind. Advances the position to the 
   * beginning of the next token
   */
  void getNextToken(std::string a, size_t& pos, std::string& str, int& kind);
};

/**
 * class for an Expression node
 */

class TxExprNode 
{
 public: 
  /**
   * constructor
   */
  TxExprNode();
  
  /**
   * constructs an exprNode with a symbol and a kind
   */
  TxExprNode(std::string, int);
  
  /**
   * converts the symbol string into a float/double value
   */
  double stringToValue();
  
  /**
   * evaluates an exprNode. Includes evaluation of all 
   * dependen nodes
   */
  double evaluate();
  
  /**
   * shows the evaluation tree 
   */
  void printEvalTree();
  
  /**
   * determines if an exprNode is constant. An expression is
   * constant, if it is constant itself, and all dependent nodes
   * are constant.
   */
  bool isConst();
  
  /**
   * if both dependent exprNodes are constant and the exprNode itself
   * is constant, evaluates the expressions and replaces the exprNode 
   * by a constant
   */
  void prune();
  
  /**
   * the symbol represented by this node
   */
  std::string symbol;
  
  /**
   * the kind of this node
   */
  int kind;
  
  /**
   * pointer to the binary operator used to evaluate this node or the
   * memory location where to find the data
   */
  union
  {
    //    BinOp binOp;
    TxBinOp * binOp;
    double* valuePtr;
  };
  
  /**
   * value represented by the node.
   */
  double value;
  
  /**
   * left argument
   */
  TxExprNode  *left;
  
  /**
   * right argument
   */
  TxExprNode  *right;
}; 


/**
 * Class for an entry in the symbol table. 
 *
 */

class TxSymbolEntry 
{
 public:
  int kind;                         // what kind is the symbol? 
  union 
  {
    TxBinOp * binOp;
    TxExprNode * expr;
    double *varPtr;                 // pointer to the variable
  };
};



/**
 * Base class for the actual symbol table. Consits of a map between
 * the symbol and a TxSymbolEntry. 
 *
 */

class TxSymbolTable
{
 public:
  /**
   * Add an operator to the symbol table
   */
  void addOperator(std::string a, TxBinOp * op);
  
  
  
  /**
   * Add an operator to the symbol table
   */
  void addExpression(std::string a, TxExprNode * op);
  
  
  /**
   * add a variable to the symbol table
   */
  void addVariable(std::string a, double* v);
  
  /**
   * get the operator reference (function pointer) for a given symbol
   */
  TxBinOp * getOperatorRef(std::string a);
  
  /**
   * 
   */
  TxExprNode * getExpressionRef(std::string a);
  
  /**
   * get the variable reference for a given symbol
   */
  double* getVariableRef(std::string a);
  
 private:
  /**
   * the map representing the actual symbol table
   */
  std::map<std::string, TxSymbolEntry  > symbols;
  
};


/**
 * parser class
 * 
 * The parser takes a expression and splits it into tokens (done
 * by the class 'lex').  The tokens are stored in a linear list
 * for simple handling. Tokens are indentified to be ALPHA symbols
 * or NUMBER symbols.
 * 
 * The token list is then parsed, based on the grammar for
 * mathematical expressions and a logical evaluation tree is
 * built (Data is not actually copied, so everything remains in
 * the list. However, for each element, pointers to the left-
 * and right-argument are set up.
 * 
 * In a next step, the symbol table is searched for all non-number
 * symbols. If an operator symbol is found, a function pointer to
 * the corresponding function is inserted in the node. Otherwiese,
 * it is assumed that the symbol is a varible. If no entry for the
 * variable exists in the symbol table, a new entry is created. The
 * space provide in the ExprNode is used as storage for the
 * variable and a reference to this piece of data is inserted in
 * the symbol table. If the variable is found in the symbol table,
 * the reference is set to the value in the symbol table.
 * 
 */

class TxParser 
{
  public:
  
  /**
   * concstructor
   */
  TxParser();
  
  /** 
   * constructs a parser for a given expression
   */
  TxParser(std::string a);
  
  /**
   * descructor, removes the parse tree
   */
  ~TxParser();
  
  /**
   * builds a parser for a given expression
   */
  void setExpression(std::string a);
  
  /**
   * assigns a value to a symbol/variable used in an expression
   */
  void assignValue(std::string a, double val);
  
  /**
   * crudely prints the list of tokens
   */
  void printTokenList();
  
  /**
   * evaluates the parse tree
   */
  double evaluate() const;
  
  /**
   * returns a pointer to symbol/variable used in expressions
   */
  double* getValuePtr(std::string a);
  
 private:
  
  /**
   * parses an expression. 
   * expr = term (['+'|'-'] term )*
   */
  TxExprNode * parseExpr();
  
  /**
   * parses a term
   * term = fact (['*'|'/'] fact)*
   */
  TxExprNode * parseTerm();
  
  /**
   * parses a fact
   * fact = symb ('^' symb)*
   */
  TxExprNode * parseFact();
  
  /**
   * parses a symbol
   * symb = ('-'|'+') fact
   *      = '(' expr ')'
   *      = alphaTerm ('(') Expr ')' )
   *      = number
   */
  TxExprNode * parseSymb();
  
  /** 
   * builds a linear list of ExprNodes based on the input tokens
   */
  void buildTokenList(std::string a);
  
  /**
   * transforms the list structure in an evaluation tree
   */
  void buildEvalTree();
  
  /**
   * builds the default symbol table, setting up the predefined symbols
   * liken 'sin', 'cos', etc
   */
  void buildDefSymbTable();
  
  /**
   * traverses the tree and finds the undefined symbols. Assigns space for 
   * the variables
   */
  void buildSymbTable();
  
  /** 
   * prunes the evaluation tree by constatnt subexpression elimination
   */
  void pruneTree();
  
  TxBinOpPlus   opPlus;
  TxBinOpMinus  opMinus;
  TxBinOpTimes  opTimes;
  TxBinOpDiv    opDiv;
  TxBinOpPow    opPow;
  TxBinOpInv    opInv;
  TxBinOpSin    opSin;
  TxBinOpCos    opCos;
  TxBinOpTan    opTan;
  TxBinOpExp    opExp;
  TxBinOpAsin   opAsin;
  TxBinOpAcos   opAcos;
  TxBinOpAtan   opAtan;
  TxBinOpLn     opLn;
  TxBinOpH      opH;
  TxBinOpAbs    opAbs;
  TxBinOpSqrt   opSqrt;
  TxBinOpj0	opj0;
  TxBinOpj1	opj1;
  
  
  /**
   * list of the ExprNodes, holding the tokens 
   */
  std::list<TxExprNode *> l;
  
  /**
   * iterator over the exprNode list. Used when parsing the expression
   */
  std::list< TxExprNode * >::iterator i;
  
  /**
   * root of the evaluation tree
   */
  TxExprNode * expRoot;
  
  /**
   * the symbol table
   */
  TxSymbolTable  symbolTable;
};
#endif
