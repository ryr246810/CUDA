//-------------------------------------------------------------------
//
// File:        TxStrManip.h
//
// Purpose:     Interface of some utilities for dealing with strings and streams.
//
// Version:     $Id: TxStrManip.h 62 2006-09-18 20:19:27Z yew $
//
// Copyright 1996-2001, Tech-X Corporation
//
//-------------------------------------------------------------------

#ifndef TX_STR_MANIP_H
#define TX_STR_MANIP_H

#ifdef _WIN32
// #pragma warning ( disable: 4786)
// innocent pragma to keep the VC++ compiler quiet
// about the too long names in STL headers.
#endif

// std includes
#include <string>
#include <map>
#include <sstream>

// txbase includes
#include <TxDebugExcept.h>
#include <TxBase.h>
/**
 * Remove the white space in a string
 *
 * @param s the string to act upon
 *
 * @return the new string
 */
TXBASE_API std::string removeWhiteSpace(const std::string& s);

/**
 * Remove the white space at the ends of a string
 *
 * @param s the string to act upon
 *
 * @return the new string
 */
TXBASE_API std::string removeEndWhiteSpace(const std::string& s);

/**
 * Replace occurrences of one substring by another
 *
 * @param s the string to act upon
 * @param oldstr the substring to remove
 * @param newstr the substring to insert
 *
 * @return the new string
 */
TXBASE_API std::string replaceSubstring(std::string& s, const std::string& oldstr, const std::string& newstr);

/**
 * Replace occurrences of one name by another.  A name is a substring
 * that before and after it is not a letter, number, or _
 *
 * @param s the string to act upon
 * @param oldstr the name to remove
 * @param newstr the name to insert
 *
 * @return the new string
 */
TXBASE_API std::string replaceName(std::string& s, const std::string& oldstr, const std::string& newstr);

/**
 * Find the closing paren starting at some point
 *
 * @param s the string to find the closing paren of
 * @param b where to start looking
 *
 * @return index of the closing paren, length if not found
 */
TXBASE_API size_t findClosingParen(const std::string& s, size_t b);

/**
 * Find a unique name for a given map based on a given name.
 *
 * @param nm the name to start with
 * @param mp the map to look in
 *
 * @return a unique name
 */
template <class T> TXBASE_API inline  std::string uniqueName(const std::string& nm, const std::map< std::string, T, std::less<std::string> >& mp)
{
  std::string res = nm;
  for (size_t i=0; i<10000; ++i) 
    {
      typename std::map< std::string, T, std::less<std::string> >::const_iterator iter = mp.find(res);
      if (iter == mp.end()) return res;

      std::ostringstream oss;
      oss << nm << "_" << i;
      res = oss.str();
      
    }
  res = "Unique name not found for ";
  res += nm;
  throw TxDebugExcept(res);
}

/**
 * Return the number of characters that have been read
 * from the stream
 *
 * @param sstr the stringstream.
 *
 * @return the number of characters read or -1 if at end.
 */

TXBASE_API int getReadChars(std::stringstream& sstr);

#endif   // TX_STR_MANIP_H
