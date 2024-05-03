//--------------------------------------------------------------------
//
// File:	TxDblFormula.cpp
//
// Purpose:	Equation parsing implementation
//
// Version:	$Id: TxDblFormula.cpp 62 2006-09-18 20:19:27Z yew $
//
// Copyright 1996-2001, Tech-X Corporation
//
//--------------------------------------------------------------------

#ifdef _WIN32
// #pragma warning ( disable: 4786)  
// innocent pragma to keep the VC++ compiler quiet
// about the too long names in STL headers.                                     
#endif

//  system includes
#include <math.h>
#include <stdlib.h>

// tx includes
#include <TxDblFormula.h>
#include <TxStrManip.h>
#include <TxDblFormulaList.h>
#include <TxBessel.h>

using namespace std;
using namespace Bess;


TxDblFormula::TxDblFormula()
{
  // Initialize data and functions
  val = 0.;
  if (!functions.size()) setupFunctions();
  
  // Set up the ThroughStreams
  outThruStrmOwner = true;
  errThruStrmOwner = true;
  txout = new TxThroughStream(std::cout, TxThroughStream::TX_ALL);
  txerr = new TxThroughStream(std::cerr, TxThroughStream::TX_ALL);
}

TxDblFormula::TxDblFormula(const TxDblFormula& fm)
  : TxDefinition(fm) 
{
  // Initialize data and functions
  val = fm.val;
  lastName = fm.lastName;
  
  // Set up the ThroughStreams
  outThruStrmOwner = false;
  errThruStrmOwner = false;
  txout = fm.txout;
  txerr = fm.txerr;
}

TxDblFormula::TxDblFormula(const std::string& nm, const std::string& mn)
  : TxDefinition(nm, mn) 
{
  // std::cerr << "TxDblFormula::TxDblFormula invoked." << std::endl;
  
  // Remove white space
  setMeaning(removeWhiteSpace(getMeaning()));
  // std::cerr << "TxDblFormula::TxDblFormula white space removed." << std::endl;
  
  // Initialize data and functions
  val = 0.;
  setupFunctions();
  // std::cerr << "TxDblFormula::TxDblFormula functions set up." << std::endl;
  
  // Set up the ThroughStreams
  outThruStrmOwner = true;
  errThruStrmOwner = true;
  txout = new TxThroughStream(std::cout, TxThroughStream::TX_ALL);
  txerr = new TxThroughStream(std::cerr, TxThroughStream::TX_ALL);
  
  // std::cerr << "TxDblFormula::TxDblFormula returning." << std::endl;
}

TxDblFormula& TxDblFormula::operator=(const TxDblFormula& fm) 
{
  if (this != &fm) {
    // Initialize data and functions
    setName(fm.getName());
    setMeaning(fm.getMeaning());
    setValue(fm.getValue());
    lastName = fm.lastName;

    // Set up the ThroughStreams
    outThruStrmOwner = false;
    errThruStrmOwner = false;
    txout = fm.txout;
    txerr = fm.txerr;
  }

  // Return reference
  return *this;
}

void TxDblFormula::setupFunctions() 
{
  if (functions.size()) return;
  
  //  Add each of the functions
  
  std::string txs("acos");
#if defined(__BCPLUSPLUS__)
  TxDblFunction facos(txs, acos);
#else
  TxDblFunction facos(txs, &acos);
#endif
  functions.push_back(facos);

  std::pair<const std::string, size_t> txpr(txs, functions.size()-1);
  functionIndices.insert(txpr);
  
  txs = "abs";
#if defined(__BCPLUSPLUS__)
  TxDblFunction myabs(txs, fabs);
#else
  TxDblFunction myabs(txs, &fabs);
#endif
  functions.push_back(myabs);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "asin";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fasin(txs, asin);
#else
  TxDblFunction fasin(txs, &asin);
#endif
  functions.push_back(fasin);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "atan";
  std::string tacos("atan");
#if defined(__BCPLUSPLUS__)
  TxDblFunction fatan(txs, atan);
#else
  TxDblFunction fatan(txs, &atan);
#endif
  functions.push_back(fatan);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "cos";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fcos(txs, cos);
#else
  TxDblFunction fcos(txs, &cos);
#endif
  functions.push_back(fcos);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "cosh";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fcosh(txs, cosh);
#else
  TxDblFunction fcosh(txs, &cosh);
#endif
  functions.push_back(fcosh);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  txs = "exp";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fexp(txs, exp);
#else
  TxDblFunction fexp(txs, &exp);
#endif
  functions.push_back(fexp);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "log";
#if defined(__BCPLUSPLUS__)
  TxDblFunction flog(txs, log);
#else
  TxDblFunction flog(txs, &log);
#endif
  functions.push_back(flog);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "log10";
#if defined(__BCPLUSPLUS__)
  TxDblFunction flog10(txs, log10);
#else
  TxDblFunction flog10(txs, &log10);
#endif
  functions.push_back(flog10);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "sin";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fsin(txs, sin);
#else
  TxDblFunction fsin(txs, &sin);
#endif
  functions.push_back(fsin);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "sinh";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fsinh(txs, sinh);
#else
  TxDblFunction fsinh(txs, &sinh);
#endif
  functions.push_back(fsinh);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "sqrt";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fsqrt(txs, sqrt);
#else
  TxDblFunction fsqrt(txs, &sqrt);
#endif
  functions.push_back(fsqrt);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "tan";
#if defined(__BCPLUSPLUS__)
  TxDblFunction ftan(txs, tan);
#else
  TxDblFunction ftan(txs, &tan);
#endif
  functions.push_back(ftan);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );
  
  txs = "tanh";
#if defined(__BCPLUSPLUS__)
  TxDblFunction ftanh(txs, tanh);
#else
  TxDblFunction ftanh(txs, &tanh);
#endif
  functions.push_back(ftanh);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  


  txs = "bess_j0";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fj0(txs, bess_j0);
#else
  TxDblFunction fj0(txs, &bess_j0);
#endif
  functions.push_back(fj0);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_j1";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fj1(txs, bess_j1);
#else
  TxDblFunction fj1(txs, &bess_j1);
#endif
  functions.push_back(fj1);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_y0";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fy0(txs, bess_y0);
#else
  TxDblFunction fy0(txs, &bess_y0);
#endif
  functions.push_back(fy0);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_y1";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fy1(txs, bess_y1);
#else
  TxDblFunction fy1(txs, &bess_y1);
#endif
  functions.push_back(fy1);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_i0";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fi0(txs, bess_i0);
#else
  TxDblFunction fi0(txs, &bess_i0);
#endif
  functions.push_back(fi0);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_i1";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fi1(txs, bess_i1);
#else
  TxDblFunction fi1(txs, &bess_i1);
#endif
  functions.push_back(fi1);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_k0";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fk0(txs, bess_k0);
#else
  TxDblFunction fk0(txs, &bess_k0);
#endif
  functions.push_back(fk0);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

  
  txs = "bess_k1";
#if defined(__BCPLUSPLUS__)
  TxDblFunction fk1(txs, bess_k1);
#else
  TxDblFunction fk1(txs, &bess_k1);
#endif
  functions.push_back(fk1);
  functionIndices.insert(std::pair<const std::string, size_t>(txs, functions.size()-1) );

}


int TxDblFormula::set(const std::string& txs) 
{
  // Create from a string
  // Get name
  // std::cerr << "string is '" << txs << "'" << std::endl;
  size_t eqpos = txs.find_first_of('=');
  // If no equals, set, but with vanishing value, return error
  if ( eqpos == std::string::npos ) {
    *txerr << "TxDblFormula::TxDblFormula(const std::string&): string \'" << txs << "\' does not contain \'=\'.\n";
    eqpos = txs.length();
    val = 0;
    return TXDF_EMPTY_FORMULA;
  }
  // std::cerr << "equals found at " << eqpos << std::endl;
  
  std::string s = txs.substr(0, eqpos);
  s = removeWhiteSpace(s);
  setName(s);
  
  // Get formula
  s = txs.substr(eqpos + 1, txs.length() - eqpos - 1);
  setMeaning(s);
  
  return 0;
}


void TxDblFormula::replaceSubstring(const std::string& txs1, 
				    const std::string& txs2) 
{
  std::string t = getMeaning();
  std::string res = ::replaceSubstring(t, txs1, txs2);
  setMeaning(res);
}

void TxDblFormula::replaceName(const std::string& txs1, 
			       const std::string& txs2) 
{
  std::string t = getMeaning();
  std::string res = ::replaceName(t, txs1, txs2);
  setMeaning(res);
}

int TxDblFormula::evaluate( const std::map<std::string, double, std::less<std::string> >& params, 
			    const TxDblFormulaList& formulas, 
			    size_t lastTxDblFormula)
{
#ifdef DEBUG
#define EVAL_DEBUG
#endif


  // If empty, set value to zero, but note problem
  if (!getMeaning().length()) {
    val = 0;
    return TXDF_EMPTY_FORMULA;
  }

  //  If no error, returns the number of subevaluations
  //  The only way to have just one evaluation is if this is a pure number
  
  //
  //  evaluate this formula using the formulas below index lastTxDblFormula
  //  May also need functions
  lastName = "";
  lastBegin = getMeaning().length();
  lastEnd = getMeaning().length();
  std::string nameStr;
  
  int numEvals=0;
  
  //
  //  Now begin find formula
  size_t begin=0, end=0;
  int negate, iSubForm=0;
  
  //  Some local storage for the numbers
  std::vector<char>	sepChars;	//  The separating chars
  std::vector<double>	locValues;	//  Found values
  
#ifdef EVAL_DEBUG
  *txerr << std::endl;
  *txerr << "Evaluating '" << getName() << " = " << getMeaning() << "', maxEvals = " << maxEvals << std::endl;
#endif

  while ( end < getMeaning().length() ) {
    negate = 1;	//  That is, no negation
    maxEvals--;
    if ( maxEvals < 0 ) {
#ifdef DEBUG
      *txerr << "TxDblFormula::evaluate: too many evaluations getMeaning() = \'" << getMeaning() << "\'.\n";
#endif
      return TXDF_TOO_MANY_EVALS;
    }
    
    // The premise is that this while statement begins just
    // after a number or variable has been found, or a 
    // subformula evaluated, or a function evaluated
    //
    // Check for leading minus sign
    switch ( getMeaning()[begin] ) {
    case '-':
      negate = -1;
    case '+':
      begin++;
      break;
    }
    
    //
    //  Determine next type:
    
    if ( ((getMeaning()[begin]>='0') && (getMeaning()[begin]<='9')) || (getMeaning()[begin]=='.') ) {
      //  We have a number
      double numVal;
      const char* beginPtr;
      char* endPtr;
      std::string subs = getMeaning().substr(begin);
      beginPtr = subs.c_str();
      numVal = strtod(beginPtr, &endPtr);
      end = begin + endPtr - beginPtr;
#ifdef EVAL_DEBUG
      *txerr << "Number " << numVal << " found" << std::endl;
      *txerr << "end = " << end << std::endl;
      *txerr << "meaning = " << getMeaning() << std::endl;
      *txerr << "meaning length = " << getMeaning().length() << std::endl;
#endif
      // END MODS
      if (end==begin) {
#ifdef EVAL_DEBUG
	*txerr << "TxDblFormula::evaluate: error in converting \'" <<  getMeaning() <<"\' starting from position " << begin << " to a number.\n";
#endif
	return TXDF_NUM_CONVERT_ERR;
      }
      if ( negate == -1) numVal = -numVal;
      locValues.push_back(numVal);	//  Add to list
      if ( end < getMeaning().length() ) {
	sepChars.push_back(getMeaning()[end]);
#ifdef EVAL_DEBUG
	*txerr << "added separator '" << getMeaning()[end] << "'\n";
#endif
      }
      end++;
    }  //if ( ((getMeaning()[begin]>='0') && (getMeaning()[begin]<='9')) || (getMeaning()[begin]=='.') )
    
    else if ( ((getMeaning()[begin]>='a') && (getMeaning()[begin]<='z')) 
	      || ((getMeaning()[begin]>='A') && (getMeaning()[begin]<='Z'))
	      || (getMeaning()[begin]=='_') ) {
      //  We have a variable, formula, or a function.  Find end.
      end = getMeaning().find_first_of(separators, begin);
      if ( end == std::string::npos ) end = getMeaning().length();
      nameStr = getMeaning().substr(begin, end - begin);
      nameStr = removeWhiteSpace(nameStr);
#ifdef EVAL_DEBUG
      *txerr << "nameStr is '" << nameStr << "'\n";
 #endif
      
      if ( (end < getMeaning().length()) && (getMeaning()[end] == '(' ) ) {
	// We have a function
#ifdef EVAL_DEBUG
	*txerr << "TxDblFormula::evaluate: function '" << nameStr << "' found.\n";
#endif
        std::map< std::string, size_t, std::less<std::string> >::iterator iter = functionIndices.find(nameStr);
	if (iter == functionIndices.end() ) {
#ifdef EVAL_DEBUG
	  *txerr << "TxDblFormula::evaluate: function '" << nameStr << "' not implemented.\n";
#endif
	  lastName = nameStr;
	  return TXDF_FUNC_NOT_DEF;
	}
        int ifunc = iter->second;
	// Get  subformula
	begin = end + 1;
        end = findClosingParen(getMeaning(), begin);
        if ( end == std::string::npos) end = getMeaning().length();
        if ( end >= getMeaning().length() ) {
#ifdef EVAL_DEBUG
	  *txerr << "TxDblFormula::evaluate: parentheses not balanced in \'" << getMeaning() << "\'.\n";
#endif
	  return TXDF_PAREN_UNBAL_ERR;
        }

        std::string argName = getName();
        argName += "_";
        std::ostringstream oss;
        oss << iSubForm;
        argName += oss.str();
	
#ifdef EVAL_DEBUG
	*txerr << "argName is '" << argName << "'\n";
#endif
	
        std::string subMeaning = getMeaning().substr(begin, end-begin);
	
        TxDblFormula subTxDblFormula(argName, subMeaning); //  Created if needed
#ifdef EVAL_DEBUG
	*txerr << "TxDblFormula::evaluate: creating formula '" << subTxDblFormula << "'" << std::endl;
#endif
        int evals = subTxDblFormula.evaluate(params, formulas, lastTxDblFormula);
	
	// If error, propagate up
        if (evals<0) {
#ifdef EVAL_DEBUG
          *txerr << "TxDblFormula::evaluate: formula \'" << subMeaning << " not defined.  Error is ";
#endif
	  switch (evals) {
	  case TXDF_NOT_IN_LIST:
	    lastName = subTxDblFormula.getLastName();
#ifdef EVAL_DEBUG
	    *txerr << "formula not in list.\n";
	    *txerr << "lastName set to '" << lastName << "'\n";
#endif
	    break;
	  default:
#ifdef EVAL_DEBUG
	    *txerr << "#" << evals << ".\n";
#endif
	    break;
          }
          return evals;
        }
	
        double u = subTxDblFormula.getValue();
        double f = functions[ifunc].evaluate(u);  // evalutate funcion with the caculated variable
        if ( negate == -1) f = -f;
        locValues.push_back(f);	//  Add to vectors
        end++;
#ifdef EVAL_DEBUG
	*txerr << "inserting value " << f << std::endl;
#endif
        numEvals += (evals + 1);
      }  //if ( (end < getMeaning().length()) && (getMeaning()[end] == '(' ) ) 

      else {
	// We have a variable or a formula
#ifdef EVAL_DEBUG
	*txerr << "Looking for formula '" << nameStr << "' in";
	formulas.printEvalResults();
#endif
	size_t iform = formulas.getIndex(nameStr);
	
	if ( iform < formulas.getNumFormulas() ) {   // a formula
#ifdef EVAL_DEBUG
	  *txerr << "formula or variable named '" << nameStr << "' found\n";
#endif
	  // We have a formula
	  if ( iform >= lastTxDblFormula) {
	    lastName = nameStr; lastBegin = begin; lastEnd = end - 1;
#ifdef EVAL_DEBUG
	    *txerr << "The formula '" << *this << "' cannot be evaluated yet.\n";
	    *txerr << "  lastName = '" << lastName << "', iform = " << iform << ", lastTxDblFormula = " << lastTxDblFormula << std::endl;
#endif
	    return TXDF_CANNOT_EVALUATE;
	  }
	  // Add to vectors
	  locValues.push_back(negate*formulas[iform].getValue());  
#ifdef EVAL_DEBUG
	  *txerr << "  formula named '" << formulas[iform].getName() << "' has value = " << formulas[iform].getValue() << std::endl;
#endif
	}
	
	else {  // a variable
	  std::map< std::string, double, std::less<std::string> >::const_iterator prmiter = params.find(nameStr);
	  if ( prmiter != params.end() ) {
#ifdef EVAL_DEBUG
	    *txerr << "parameter '" << nameStr << " has value " << prmiter->second << std::endl;
#endif
	    // We have a variable
	    locValues.push_back(negate*prmiter->second); //  Add to vectors
#ifdef EVAL_DEBUG
	    *txerr << "inserting value " << negate*prmiter->second << std::endl;
#endif
	  }
	  else {
	    // Name is in neither list
	    lastName = nameStr; lastBegin = begin; lastEnd = end - 1;
#ifdef EVAL_DEBUG
	    *txerr << "TxDblFormula::evaluate: '" << lastName << "' is neither parameter nor formula.\n";
#endif
	    return TXDF_NOT_IN_LIST;
	  }
	}
      }   //if ( (end < getMeaning().length()) && (getMeaning()[end] == '(' ) ) else-------->

      if ( end < getMeaning().length() ) {
	sepChars.push_back(getMeaning()[end]);
#ifdef EVAL_DEBUG
	*txerr << "appending separating char " << getMeaning()[end] << std::endl;
#endif
      }
      numEvals++;
      end++;
    }  //else if ( ((getMeaning()[begin]>='a') && (getMeaning()[begin]<='z')) || ((getMeaning()[begin]>='A') && (getMeaning()[begin]<='Z')) || (getMeaning()[begin]=='_') )
    
    else if (getMeaning()[begin]=='(') {      //  We have a subformula
       end = findClosingParen(getMeaning(), begin+1);
      if ( end >= getMeaning().length() ) {
#ifdef EVAL_DEBUG
	*txerr << "TxDblFormula::evaluate: parentheses not balanced in \'" << getMeaning() << "\'.\n";
#endif
	return TXDF_PAREN_UNBAL_ERR;
      }
      nameStr = getName();
      nameStr += "_";
      
      std::ostringstream ss;
      ss << iSubForm;
      nameStr += ss.str();

      std::string subMeaning = getMeaning().substr(begin+1, end-begin-1);
      TxDblFormula subTxDblFormula(nameStr, subMeaning); //  Created if needed
      int evals = subTxDblFormula.evaluate(params, formulas, lastTxDblFormula);
      
      // If error, propagate up
      if (evals<0) {
#ifdef EVAL_DEBUG
        *txerr << "TxDblFormula::evaluate: formula \'" << subMeaning << " not defined.  Error is ";
#endif
        switch (evals) {
	case TXDF_NOT_IN_LIST:
	  lastName = subTxDblFormula.getLastName();
#ifdef EVAL_DEBUG
	  *txerr << "formula not in list.\n";
#endif
	  break;
	default:
#ifdef EVAL_DEBUG
	  *txerr << "#" << evals << ".\n";
#endif
	  break;
        }
        return evals;
      }
      
      double u = subTxDblFormula.getValue();
      if ( negate == -1) u = -u;
      locValues.push_back(u);	//  Add to vectors
      end++;
      if ( end < getMeaning().length() ) sepChars.push_back(getMeaning()[end]);
      numEvals += evals;
      end++;
    } // subformula
    

    else {
      //  Unknown beginning character!
#ifdef EVAL_DEBUG
      *txerr << "TxDblFormula::evaluate: error in parse.  Character = " << getMeaning()[begin] << std::endl;
#endif
      return TXDF_UNK_BEGIN_CHAR;
    }
    
    //  In any case, next character must be arithmetic
    if ( end < getMeaning().length() ) {
      switch ( sepChars[sepChars.size()-1] ) {
      case '+':
      case '-':
      case '*':
      case '/':
      case '0':
	break;
      default:
#ifdef EVAL_DEBUG
	*txerr << "TxDblFormula::evaluate: number \'" << nameStr<< "\' not followed by arithmetic operator (+, -, *, /).\n";
#endif
	return TXDF_NUM_FOLLOW_ERR;
      }
    }
    
    begin = end;
  }
  
  //
  //  Check to make sure formula was evaluated
  if ( end < getMeaning().length() ) {
#ifdef EVAL_DEBUG
    *txerr << "TxDblFormula::evaluate: could not parse \'" << getMeaning()
	   << "\' for unknown reasons.\n";
#endif
    return TXDF_UNK_PARSE_ERR;
  }


#ifdef EVAL_DEBUG
  *txerr << "Collapsing: ";
  for (size_t i=0; i<locValues.size()-1; ++i) {
    *txerr << locValues[i] << sepChars[i];
  }
  *txerr << locValues[locValues.size()-1];
  *txerr << std::endl;
#endif
  
  //
  //  Collapse with multiplicative operators
  size_t iTo=0, iFrom;
  for (iFrom=1; iFrom<locValues.size(); iFrom++) {
    register int ifm = iFrom - 1;
    switch ( sepChars[ifm] ) {
    case '*':
      locValues[iTo] *= locValues[iFrom];
      numEvals++;
      break;
    case '/':
      locValues[iTo] /= locValues[iFrom];
      numEvals++;
      break;
    case '+':
    case '-':		//  Simply move all to iTo
      if (iFrom-iTo>1) {
	sepChars[iTo] = sepChars[ifm];
	locValues[iTo+1] = locValues[iFrom];
      }
    iTo++;
    break;
    }
  }
  
  //
  //  Collapse with additive operators
  for (iFrom=1; iFrom<=iTo; iFrom++) {
    register int ifm = iFrom - 1;
    switch ( sepChars[ifm] ) {
    case '+':
      locValues[0] += locValues[iFrom];
      numEvals++;
      break;
    case '-':
      locValues[0] -= locValues[iFrom];
      numEvals++;
      break;
    }
  }
  val = locValues[0];

#ifdef EVAL_DEBUG
  *txerr <<  "TxDblFormula::evaluate: final value (post additive collapse) for "<< getName() << " is " << getValue() << std::endl;
#endif
  
  numEvals++;	//  This evaluation
  return numEvals;
}


void TxDblFormula::setOutThroughStream(TxThroughStream& t) const {
  if (outThruStrmOwner) {
    delete txout;
    outThruStrmOwner = false;
  }
  txout = &t;
}

void TxDblFormula::setErrThroughStream(TxThroughStream& t) const {
  if (errThruStrmOwner) {
    delete txerr;
    errThruStrmOwner = false;
  }
  txerr = &t;
}

// The static variables

std::string TxDblFormula::separators = "+-*/()";
int TxDblFormula::maxEvals = 100;
std::vector< TxDblFunction >  TxDblFormula::functions;
std::map< std::string, size_t, std::less<std::string> > 
TxDblFormula::functionIndices;

