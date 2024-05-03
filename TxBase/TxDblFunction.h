//-------------------------------------------------------------------
//
// File:        TxDblFunction.h
//
// Purpose:     Class for functions - evaluate given a name
//
// Version:     $Id: TxDblFunction.h 58 2006-09-18 17:25:05Z yew $
//
// Copyright 1999-2001, Tech-X Corporation
//
//-------------------------------------------------------------------

#ifndef TX_DBL_FUNCTION_H
#define TX_DBL_FUNCTION_H

#ifdef _WIN32
// #pragma warning ( disable: 4786)  
// innocent pragma to keep the VC++ compiler quiet
// about the too long names in STL headers.                                     
#endif

// system includes
#include <assert.h>
#include <TxBase.h>
/**
 *    --  Class for functions;  given a name, evaluate it appropriately.
 * Maintains an internal list of string-functions pairs.  Given a name
 * it finds the function and evaluates it
 *
 *  Copyright 1996-2001 by Tech-X Corporation
 *
 *  @author  John R. Cary
 *
 *  @version $Id: TxDblFunction.h 58 2006-09-18 17:25:05Z yew $
 */

class TXBASE_API TxDblFunction 
{

 public:
  
  /**
   * default constructor
   */
 TxDblFunction() : name(""), function(0) {};
  
  /**
   * Construct with a specified name and function pointer
   *
   * @param nm the name of this function
   * @param fnc the funtion pointer for this function
   */
 TxDblFunction(const std::string& nm, double (*fnc)(double)) 
   : name(nm), function(fnc) {}
  
  /**
   * virtual destructor
   */
  virtual ~TxDblFunction() {}
  
  /**
   * evaluate the function
   *
   * @param db the input parameter
   *
   * @return the value of the function
   */
  double evaluate(const double& db) { 
    assert((bool)(function!=0));
    double val = (*function)(db);
    if ( val != val ) {
      std::cerr << "TxDblFunction::evaluate: error in evaluation of "<< name << '(' << db << ").\n";
    }
    return val;
  }
  
 private:
  
  /** the name of this function */
  std::string name;
  
  /** the function pointer */
  double (*function)(double);
};

#endif   // TX_DBL_FUNCTION_H
