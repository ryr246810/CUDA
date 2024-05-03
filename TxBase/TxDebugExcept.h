//
// File:	TxDebugExcept.h
//
// Purpose:	Basic exception that one can add strings to
//
// Version:	$Id: TxDebugExcept.h 127 2007-11-29 20:46:46Z cary $
//
// Copyright 1999-2001, Tech-X Corporation
//
//-----------------------------------------------------------------------

#ifndef TX_DEBUG_EXCEPT_H
#define TX_DEBUG_EXCEPT_H

// system includes
#include <stdio.h>
#include <iostream>
#include <sstream>


#ifdef _MSC_VER
#define uint32_t unsigned long
#define uint64_t unsigned long long
#else
#include<stdint.h>
#endif

// txbase includes
#include <TxBase.h>


/** Base class for exceptions that want to have simple debugging.
 *
 *  Copyright (c) 1999 by Tech-X Corporation.  All rights reserved.
 *
 *  @author  Ryan McLean
 *  @version $Id: TxDebugExcept.h 127 2007-11-29 20:46:46Z cary $
 */

class TXBASE_API TxDebugExcept
{
  
 public:
  
  /// Default constructor
  TxDebugExcept();
  
  /// Constructor taking a string.
  TxDebugExcept(const std::string&);
  
  /// Copy constructor
  TxDebugExcept(const TxDebugExcept&);
  
  /// Destructor: virtual -- All resources released implicitly
  virtual ~TxDebugExcept();
  
  /// Assign from another of the same
  TxDebugExcept& operator=(const TxDebugExcept&);

  /// Returns a string of the accumulated messages.
  std::string getMessage() const;
  
  /** Places an integer into the message.
   *  @param i the integer to add
   *  @returns the modified TxDebugExcept
   */
  TxDebugExcept& operator<<(int i);
  
  /** Places an unsigned integer into the message.
   *  @param i the unsigned integer to add
   *  @returns the modified TxDebugExcept
   */
  TxDebugExcept& operator<<(size_t i);
  
  /** Places a float into the message.
   *  @param f the float to add
   *  @returns the modified TxDebugExcept
   */
  TxDebugExcept& operator<<(float f);
  
  /** Places a double into the message.
   *  @param d the double to add
   *  @returns the modified TxDebugExcept
   */
  TxDebugExcept& operator<<(double d);
  
  /** Places a long double into the message.
   *  Not available because aCC chokes
   *  @param d the long double to add
   *  @returns the modified TxDebugExcept
   TxDebugExcept& operator<<(long double d);
  */
  
  /** Places a string into the message.
   *  @param str the string to add
   *  @returns the modified TxDebugExcept
   */
  TxDebugExcept& operator<<(std::string str);
  
 protected:
  
  /// Output stream for debug info
  mutable std::ostringstream* debugStream;
  
 private:
  
  // Prevent use
  
};

TXBASE_API std::ostream& operator<<(std::ostream& s, class TxDebugExcept& d);

#endif
