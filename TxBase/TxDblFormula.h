//-------------------------------------------------------------------
//
// File:        TxDblFormula.h
//
// Purpose:     Interface of a class that can parse equations
//
// Version:     $Id: TxDblFormula.h 59 2006-09-18 18:04:10Z yew $
//
// Copyright (c) 1996-2003, Tech-X Corporation.  All Rights Reserved.
//
//-------------------------------------------------------------------

#ifndef TX_DBL_FORMULA_H
#define TX_DBL_FORMULA_H

// system includes
#include <stdio.h>

// std includes
#include <vector>


// txbase includes
#include <TxThroughStream.h>
#include <TxBase.h>

// txmapabase includes
#include <TxDefinition.h>
#include <TxDblFunction.h>
class TxDblFormulaList;


// Error conditions
#define TXDF_NUM_CONVERT_ERR            -5
#define TXDF_NUM_FOLLOW_ERR             -6
#define TXDF_PAREN_UNBAL_ERR            -7
#define TXDF_SUBFORM_UNDEF_ERR          -8
#define TXDF_UNK_PARSE_ERR              -9
#define TXDF_UNK_BEGIN_CHAR             -10
#define TXDF_TOO_MANY_EVALS             -11
#define TXDF_CANNOT_EVALUATE            -12
#define TXDF_NOT_IN_LIST                -13
#define TXDF_FUNC_NOT_DEF               -14
#define TXDF_DUPLICATE_NAME             -15
#define TXDF_EMPTY_FORMULA              -16

/**
 *    --  Class for handling mathematical expressions.
 *
 *  The string name holds the expression.
 *  The double value contains the result evaluating the expression.
 *
 *  Copyright 1996, 1997, 1998 by Tech-X Corporation
 *
 *  @author  John R. Cary
 *
 *  @version $Id: TxDblFormula.h 59 2006-09-18 18:04:10Z yew $
 */

class TXBASE_API TxDblFormula : public TxDefinition 
{
  // friend class TxKeyedFormulaList;
  
 public:
  
  /**
   * Default constructor
   */
  TxDblFormula();
  
  /**
   * Copy constructor
   *
   * @param fm the TxDblFormula to copy from
   */
  TxDblFormula(const TxDblFormula& fm);
  
  /**
   * construct with strings specifying name and meaining of expression
   *
   * @param nm the new name
   * @param mn the meaning or forumla
   */
  TxDblFormula(const std::string& nm, const std::string& mn);
  
  /**
   * virtual destructor
   */
  virtual ~TxDblFormula() {
    if (outThruStrmOwner) delete txout;
    if (errThruStrmOwner) delete txerr;
  }
  
  /**
   * assignment operator
   *
   * @param fm the formula to assign from
   */
  TxDblFormula& operator=(const TxDblFormula& fm);
  
  /**
   * Set formula from a single string
   *
   * @param txs the string to set from.  Must contain an equals to
   * separate the name from the meaning.
   *
   * @return error - 0 for no error
   */
  int set(const std::string& txs);
  
  /**
   * Set the maximum allowed number of evaluations
   *
   * @param i the new maximum allowed number of evaluations
   */
  void setMaxEvals(int i) { maxEvals = i;}
  
  /**
   * Get the value of the formula
   */
  double getValue() const { return val;}
  
  /**
   * Replace a substring in the meaning of this formula
   *
   * s1 the old substring
   * s2 the new to use as a replacement
   */
  void replaceSubstring(const std::string&, const std::string&);
  
  /**
   * Replace a name in the meaning of this formula.  A name is a substring
   * such that before and after it is not a letter, number, or _
   *
   * s1 the old substring
   * s2 the new to use as a replacement
   */
  void replaceName(const std::string&, const std::string&);
  
  /** 
   * evaluate from a double list, associated indices, formulas,
   * and associated indices.
   *
   * @param prms map of strings to parameters to be used in the evaluation
   * @param forms set of formulas for evaluating  this formula
   * @param defForms the number of formulas defined in defForms
   */
  int evaluate( const std::map< std::string, double, std::less<std::string> >& prms,
		const TxDblFormulaList& forms, 
		size_t defForms);
  
  /**
   * set the TxThroughStream for cout
   *
   * @param t the new through stream replacing cout
   */
  virtual void setOutThroughStream(TxThroughStream& t) const;
  
  /**
   * get the TxThroughStream for cout
   *
   * @return the through stream replacing cout
   */
  virtual TxThroughStream *getOutThroughStream() const { return txout;}
  
  /**
   * set the TxThroughStream for cerr
   *
   * @param t the new through stream replacing cout
   */
  virtual void setErrThroughStream(TxThroughStream& t) const;
  
  /**
   * get the TxThroughStream for cerr
   *
   * @return the through stream replacing cerr
   */
  virtual TxThroughStream *getErrThroughStream() const {return txerr;}
  
  /** 
   * Write the formula to an output stream
   *
   * @param ostr the output stream to write the formula to
   * @param fm the formula to write
   *
   * @return the output stream
   */
  friend std::ostream& operator<< (std::ostream& ostr, const TxDblFormula &fm) {
    ostr << fm.getName() << "=" << fm.getMeaning() << "=" << fm.getValue();
    return ostr;
  }
  
  /**
   * get the last unknown name
   *
   * @return the last unknown name 
   */
  const std::string getLastName() { return lastName;}
  
 protected:
  
  /**
   * Replace the last unknown name
   *
   * @param s the string to replace it with
   void replaceLastName(const std::string& s);
  */
  
  /**
   * Set the value of this formula
   *
   * @param u the new value
   */
  void setValue(const double& u) { val = u; }
  
  /** the cout TxThroughStream */
  mutable TxThroughStream *txout;
  
  /** the cerr TxThroughStream */
  mutable TxThroughStream *txerr;
  
 private:
  
  /**
   * method for various setup procedures
   */
  void setupFunctions();
  
  /** The value of the formula */
  double val;
  
  /** Separating chars: +, -, *, /, (, ) */
  static std::string separators;
  
  /** Maximum evaluations allowed */
  static int maxEvals;
  
  /** Last string being worked on. */
  std::string lastName;
  
  /** Location in formula of start for lastName */
  int lastBegin;
  
  /** Location in formula of end for lastName */
  int lastEnd;
  
  /** Whether the out TxThroughStream is owned by this class **/
  mutable bool outThruStrmOwner;
  
  /** Whether the err TxThroughStream is owned by this class **/
  mutable bool errThruStrmOwner;
  
  /**
   * Functions known by the formulas
   */
  static std::vector< TxDblFunction > functions;
  
  /**
   * Lookup list for finding known functions by index
   */
  static std::map< std::string, size_t, std::less<std::string> > functionIndices;
  
};

#endif   // TX_DBL_FORMULA_H
