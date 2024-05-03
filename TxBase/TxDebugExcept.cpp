//-----------------------------------------------------------------------
//
// File:    TxDebugExcept.cpp
//
// Purpose: Base class for providing debugging error information
//
// Version: $Id: TxDebugExcept.cpp 138 2008-01-16 21:46:50Z cary $
//
// Copyright (c) 1999 by Tech-X Corporation.  All rights reserved.
//
//-----------------------------------------------------------------------

// txbase includes
#include <TxDebugExcept.h>


// default constructor
TxDebugExcept::TxDebugExcept()
{
  debugStream = new std::ostringstream;
  *debugStream << "\n";
}

// constructor taking a string
TxDebugExcept::TxDebugExcept(const std::string& str) 
{
  debugStream = new std::ostringstream;
  *debugStream << str;
}

// Copy constructor
TxDebugExcept::TxDebugExcept(const TxDebugExcept& d) 
{
  std::string s( d.debugStream->str());
  debugStream = new std::ostringstream;

  *debugStream << s;
  delete d.debugStream;

  d.debugStream = new std::ostringstream;

  *(d.debugStream) << s;
}

// Destructor: virtual -- All resources released implicitly
TxDebugExcept::~TxDebugExcept()
{
  delete debugStream;
}

// Copy constructor
TxDebugExcept& TxDebugExcept::operator=(const TxDebugExcept& d) 
{
  if ( this == &d ) return *this;
// Get string
  std::string s( d.debugStream->str());

// remove old stream of this object and replace with string
  delete debugStream;
  debugStream = new std::ostringstream;
  *debugStream << s;
// Remove old stream of assigned to and replace with string
  delete d.debugStream;
  d.debugStream = new std::ostringstream;
  *(d.debugStream) << s;
  return *this;
}

// Returns a string of the accumulated messages.
std::string TxDebugExcept::getMessage() const 
{
  // ostrstream& dS( const_cast<ostrstream&>( debugStream ) );
  // std::string s( dS.str(), dS.pcount() );
  // dS.rdbuf()->freeze( 0 );
  std::string s( debugStream->str());

  delete debugStream;

  debugStream = new std::ostringstream;
  *debugStream << s;
  return s;
}


//
TxDebugExcept& TxDebugExcept::operator<<(int i)
{
  *debugStream << i;
  return *this;
}

//
TxDebugExcept& TxDebugExcept::operator<<(size_t s) 
{
  *debugStream << s;
  return *this;
}

//
TxDebugExcept& TxDebugExcept::operator<<(float f)
{
  *debugStream << f;
  return *this;
}

//
TxDebugExcept& TxDebugExcept::operator<<(double d) 
{
  *debugStream << d;
  return *this;
}

/* aCC chokes
//
TxDebugExcept& TxDebugExcept::operator<<(long double ld) 
{
  *debugStream << ld;
  return *this;
}
*/

//
TxDebugExcept& TxDebugExcept::operator<<(std::string str) 
{
  *debugStream << str;
  return *this;
}

std::ostream& operator<<(std::ostream& s, class TxDebugExcept& d)
{
  return s << d.getMessage();
}

