//-------------------------------------------------------------------
//
// File:        TxDefinition.h
//
// Purpose:     pair of strings
//
// Version:     $Id: TxDefinition.h 60 2006-09-18 18:26:43Z yew $
//
// Copyright 1996-2001, Tech-X Corporation
//
//-------------------------------------------------------------------

#ifndef TX_DEFINITION_H
#define TX_DEFINITION_H



#include <iostream>
#include <string>

#include "TxBase.h"
/**
 *    --  Class relating two strings.
 *
 *  The first string is the name.  The second string is the definition, or
 *  the meaning associated with that name.
 *
 *  Copyright 1996, 1997, 1998 by Tech-X Corporation
 *
 *  @version $Id: TxDefinition.h 60 2006-09-18 18:26:43Z yew $
 */

class TXBASE_API TxDefinition 
{
 public:
  
  /**
   * Default constructor
   */
  TxDefinition() {}
  
  /**
   * Construct with a specified name
   *
   * @param nm the name for this definition
   */
  TxDefinition(const std::string& nm) {
    name = nm;
  }
  
  /**
   * Copy constructor
   *
   * @param df the definition to copy from
   */
  TxDefinition(const TxDefinition& df) {
    name = df.getName();
    meaning = df.getMeaning();
  }
  
  /**
   * Construct with a specified name and meaning
   *
   * @param nm the name
   * @param mn the meaning
   */
  TxDefinition(const std::string& nm, const std::string& mn) {
    name = nm;
    meaning = mn;
  }
  
  /**
   * virtual destructor
   */
  virtual ~TxDefinition() {}
  
  /**
   * Set the name of this definition
   *
   * @param nm the new name
   */
  void setName(const std::string& nm) { 
    name = nm;
  }
  
  /**
   * Get the name of this definition
   *
   * @return the name of this definition
   */
  std::string getName() const { 
    return name;
  }
  
  /**
   * set the meaning string
   *
   * @param mn the new meaning string
   */
  void setMeaning(const std::string& mn) { 
    meaning = mn;}
  
  /**
   * get the meaning string
   *
   * @return the meaning string
   */
  std::string getMeaning() const { 
    return meaning;
  }
  
  /**
   * For ordering
   */
  bool operator<(const TxDefinition& df) const { 
    return name < df.getName();
  }
  
  /**
   * Friend operator for ostream output
   *
   * @param ostr the output stream to write to
   * @param df the definition to write
   *
   * @return the ostream
   */
  friend std::ostream& operator<< (std::ostream& ostr, const TxDefinition& df) {
    ostr << df.getName() << " = " << df.getMeaning();
    return ostr;
  }
  
 private:
  
  /** The name for this definition */
  std::string name;
  
  /** The meaning for this definition */
  std::string meaning;
  
};

#endif   // TX_DEFINITION_H
