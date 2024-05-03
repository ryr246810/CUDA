//-------------------------------------------------------------------
//
// File:        TxDblFormulaList.h
//
// Purpose:     Interface for a list of formulas that one can find by
//		key.
//
// Version:     $Id: TxDblFormulaList.h 58 2006-09-18 17:25:05Z yew $
//
// Copyright (c) 1996-2003, Tech-X Corporation.  All Rights Reserved.
//
//-------------------------------------------------------------------

#ifndef TX_DBL_FORMULA_LIST_H
#define TX_DBL_FORMULA_LIST_H

#ifdef _WIN32
// #pragma warning ( disable: 4786)  
// #pragma warning ( disable: 4290) 
// innocent pragma to keep the VC++ compiler quiet
// about the too long names in STL headers.                                     
#endif

// txbase
#include <TxDebugExcept.h>
#include <TxThroughStream.h>
#include <TxBase.h>

// txmapabase includes
#include <TxDblFormula.h>

/**
 *    --  Keyed list of TxDblFormula objects.
 *
 *  List of formulas that know how to evaluate themselves
 *  in terms of each other and a list of named parameters.
 *
 *  Copyright 1996-2001 by Tech-X Corporation
 *
 *  @author  John R. Cary
 *
 *  @version $Id: TxDblFormulaList.h 58 2006-09-18 17:25:05Z yew $
 */

class TXBASE_API TxDblFormulaList 
{
  
 public:
  
  /**
   * Default constructor
   */
  TxDblFormulaList() {
    definedFormulas = 0;
    params = &mtparams;
    txout = new TxThroughStream(std::cout, TxThroughStream::TX_ALL);
    outThruStrmOwner = true;
    txerr = new TxThroughStream(std::cerr, TxThroughStream::TX_ALL);
    errThruStrmOwner = true;
  }
  
  /**
   * virtual destructor
   */
  virtual ~TxDblFormulaList() {
    if (outThruStrmOwner) delete txout;
    if (errThruStrmOwner) delete txerr;
  }
  
  /**
   * Set the list of parameters to aid in evaluation
   *
   * @param p the parameters to use in the evaluation.  Not a handoff.
   */
  virtual void setParams(const std::map< std::string, double, std::less<std::string> >& p) {
    params = &p;
  }
  
  /**
   * set the TxThroughStream for std::cout
   *
   * @param t the new through stream replacing std::cout
   */
  virtual void setOutThroughStream(TxThroughStream& t) {
    if (outThruStrmOwner) {
      delete txout;
      outThruStrmOwner = false;
    }
    txout = &t;
  }

  /**
   * set the TxThroughStream for std::cerr
   *
   * @param t the new through stream replacing std::cerr
   */
  virtual void setErrThroughStream(TxThroughStream& t) {
    if (errThruStrmOwner) {
      delete txerr;
      errThruStrmOwner = false;
    }
    txerr = &t;
  }
  
  /**
   * Insert a new formula into the list
   * 
   * @param tf the formula
   * @param pl the place to insert this formula
   */
  void insert(const TxDblFormula& tf, size_t pl) ;
  
  /**
   * Insert a new, defined formula into the list
   * 
   * @param tf the formula
   * @param pl the place to insert this formula
   */
  void insertDefined(const TxDblFormula& tf, size_t pl) ;
  
  /**
   * Append a new formula to the list
   * @param tf the formula
   */
  void append(const TxDblFormula& tf) ;
  
  /**
   * Append a new defined formula to the list
   * @param tf the formula
   */
  void appendDefined(const TxDblFormula& tf) ;
  
  /**
   * Remove a formula of a given name
   *
   * @param nm name of the formula to remove
   */
  void remove(const std::string& nm);
  
  /**
   * Remove all of the formulas
   *
   */
  void removeAll() {
    definedFormulas = 0;
    formulas.clear();
    formulaMap.clear();
  }
  

 public:
  void getResult(std::map< std::string, double, std::less<std::string> >& resultData);



  /**
   * Get the index of a formula by name
   *
   * @param nm the name of the formula to get
   *
   * @return the index, number of formulas if not found.
   */
  size_t getIndex(const std::string& nm) const;
  
  /**
   * Get a formula by name
   *
   * @param nm the name of the formula to get
   *
   * @return the the formula, empty if none
   */
  TxDblFormula operator[](const std::string& nm) const;
  
  /**
   * Get a formula by index
   *
   * @param i the index of the formula to get
   *
   * @return the the formula, empty if none
   */
  TxDblFormula operator[](size_t i) const;
  
  /**
   * Get the number of defined formulas
   *
   * @return the number of defined formulas
   */
  size_t getNumFormulas() const { return formulas.size();}
  
  /**
   * Get the number of defined formulas
   *
   * @return the number of defined formulas
   */
  size_t getNumDefinedFormulas() const { return definedFormulas;}
  
  /**
   * Set the number of defined formulas to zero, so all must be re-evaluated
   */
  void reset() { 
    definedFormulas = 0;
  }
  
  /**
   *
   * Return the number of redefined names
   *
   * @return the number of redefined names
   */
  size_t getNumRedefinedNames() const { return redefinedNames.size();}
  
  /**
   * get the new name of a redefined name
   *
   * @param nm the old name.
   *
   * @return the new name.  Empty if not defined.
   */
  std::string getRedefinedNewName(const std::string& nm) const; 
  
  /**
   * Replace a redefined name
   *
   * @param nm the name now known
   */
  void replaceRedefinedName(const std::string& nm);
  
  /**
   * Replace all of the names that have now been redefined
   */
  void replaceRedefinedNames();
  
  /**
   * Insert a new redefinition.
   *
   * @param s1 the old name
   * @param s2 the name transferred to
   */
  void insertRedefinition(const std::string& s1, const std::string& s2);
  
  /** Find a name in neither list 
   * 
   * @param nm the name to start from
   *
   * @return a name based on nm but not in either list
   */
  std::string getUniqueName(const std::string& nm) const;
  
  /**
   * evaluate all formula as possible given a map of parameters
   *
   * @return the number of formulas not evaluated
   */
  int evaluate();
  
  /**
   * Print the evaluated results
   *
   */
  void printEvalResults() const;
  
  /**
   * print all undefined names
   *
   */
  void printUndefined() const;
  
  /**
   * print all redefined names
   *
   */
  void printRedefined() const;
  
  /**
   * Check the uniqueness of a name
   *
   * @param nm the name to see if this is unique;
   *
   * @return whether unique
   */
  bool isUnique(const std::string& nm) const ;
  
 private:
  
  /**
   * Ensure not used
   */
  TxDblFormulaList(const TxDblFormulaList&);
  
  /**
   * Ensure not used
   */
  TxDblFormulaList& operator=(const TxDblFormulaList&);
  
  /** Parameters used for evaluation */
  const std::map< std::string, double, std::less<std::string> >* params;
  
  /** Empty parameters */
  std::map< std::string, double, std::less<std::string> > mtparams;
  
  /** A map from a formula name to the formula's index */
  std::map<std::string, size_t, std::less<std::string> > formulaMap;
  
  /** A vector containing the formulas */
  std::vector<TxDblFormula> formulas;
  
  /** Number of defined formulas */
  size_t definedFormulas;
  
  /** List of needed formulas, by name as appeared, then by new given name */
  std::map<std::string, std::string, std::less<std::string> > redefinedNames;	
  
  /** the out TxThroughStream */
  mutable TxThroughStream *txout;
  
  /** Whether the out TxThroughStream is owned by this class **/
  mutable bool outThruStrmOwner;
  
  /** the err TxThroughStream */
  mutable TxThroughStream *txerr;
  
  /** Whether the err TxThroughStream is owned by this class **/
  mutable bool errThruStrmOwner;
};

#endif   // TX_DBL_FORMULA_LIST_H
