//-----------------------------------------------------------------------
//
// File:    TxThroughStream.cpp
//
// Purpose: Class for connecting multiple ostreams through a single ostream.
//
// Version: $Id: TxThroughStream.cpp 62 2006-09-18 20:19:27Z yew $
//
// Copyright 1999-2001, Tech-X Corporation.  All rights reserved.
//
//-----------------------------------------------------------------------

// To use this header, you configure system must first determine
// the values of TIME_WITH_SYS_TIME and TM_IN_SYS_TIME.

#include "TxThroughStream.h"

#define TIME_WITH_SYS_TIME 1
// Unix includes
#ifdef TIME_WITH_SYS_TIME
  #include <time.h>
  #include <sys/time.h>
#else
  #ifdef TM_IN_SYS_TIME
    #include <sys/time.h>
  #else
    #include <time.h>
  #endif
#endif

// txbase includes

// Default constructor

TxThroughStream::TxThroughStream() :  
#ifdef _WIN32
  std::basic_ostream <char, std::char_traits <char> >(this) {
#else
  std::ostream(this) {
#endif

  state = TX_ALL;
}

TxThroughStream::TxThroughStream(std::ostream &os, txfilter f ) : 
#ifdef _WIN32
  std::basic_ostream <char, std::char_traits<char> >(this) {
#else
  std::ostream(this) {
#endif
  state = f;
  attachStream(os, f);
}


// Virtual destructor
TxThroughStream::~TxThroughStream() {
  sync();
} 

void TxThroughStream::detachStream(std::ostream &s) {

// Find the element to be removed
  std::map<std::ostream*, txfilter>::iterator iter;
  iter = strStateMap.find(&s);
  if (iter == strStateMap.end() ) return;

// Erase this element
  strStateMap.erase(iter);

}

int TxThroughStream::sync () { 
  int n = pptr () - pbase (); 
  return (n && output ( pbase (), n) != n) ? EOF : 0; 
} 

// Use overflow to do something.
int TxThroughStream::overflow(int ch) {
  int n = pptr () - pbase (); 
  if (n && sync ()) 
    return EOF; 
  if (ch != EOF) { 
      char cbuf[1]; 
      cbuf[0] = ch; 
      if (output ( cbuf, 1) != 1) 
	return EOF; 
    } 
  pbump (-n); // Reset pptr(). 
  return 0; 
}

// Helper typedefs
typedef std::ostream*  ostrPtr;
typedef std::map<ostrPtr, TxThroughStream::txfilter>::const_iterator MI;

int TxThroughStream::output (char* text, int length) {
  for (MI p = strStateMap.begin(); p != strStateMap.end(); ++p) {
    if (p->second & state)  {
      for (int i=0; i<length; i++) {
        if (text[i] != '\n')   // faster to do the most common case first.
	  *(p->first) << text[i]; 
        else  
	  *(p->first) << std::endl;
      }
    }                  
  }
  return length;
}


TxThroughStream& TxThroughStream::printTime() { 
  return *this;
}


