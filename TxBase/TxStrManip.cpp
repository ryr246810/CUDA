//--------------------------------------------------------------------
//
// File:	TxStrManip.cpp
//
// Purpose:	String utilities
//
// Version:	$Id: TxStrManip.cpp 62 2006-09-18 20:19:27Z yew $
//
// Copyright 1996-2001, Tech-X Corporation
//
//--------------------------------------------------------------------

// base includes
#include <TxStrManip.h>

// Remove the white space in a string
std::string removeWhiteSpace(const std::string& s) 
{
  std::string res;
  size_t ibegin = 0, iend = 0;
  while (iend<s.length()) 
    {
      ibegin = iend;
      iend = s.find_first_of(" \t\n", ibegin);
      if (iend == std::string::npos) iend = s.length();
      res += s.substr(ibegin, iend - ibegin);
      iend ++;
  }
  return res;
}

// Remove the white space at the ends of a string

std::string removeEndWhiteSpace(const std::string& s)
{
  size_t ibegin = s.find_first_not_of(" \t\n");
  size_t iend = s.find_last_not_of(" \t\n");
  std::string res = s.substr(ibegin, iend + 1 - ibegin);
  return res;
}

// Replace occurrences of one substring by another
std::string replaceSubstring(std::string& s, const std::string& oldstr, const std::string& newstr) 
{
  std::string res;
  size_t ibegin = 0;
  size_t iend = 0;
  // cerr << "replaceSubstring: entered" << endl;
  while (iend<s.length())
    {
      ibegin = iend;
      iend = s.find(oldstr, ibegin);
      // cerr << "replaceSubstring: " << oldstr << " found at " << iend << endl;
      if (iend == std::string::npos) iend = s.length();
      res += s.substr(ibegin, iend - ibegin);
      if (iend < s.length()) res += newstr;
      iend += oldstr.length();
    }
  return res;
}

// Replace occurrences of one name by another
std::string replaceName(std::string& s, const std::string& oldstr, const std::string& newstr)
{
  std::string res;
  size_t ibegin = 0, iend = 0;
  while (iend<s.length()) {
    ibegin = iend;
    iend = s.find(oldstr, ibegin);
    if (iend == std::string::npos) iend = s.length();
    res += s.substr(ibegin, iend - ibegin);
    char tchar;
    if (iend < s.length()){
      // Ensure that char before is not a letter, number or _
      if (iend != 0) {
	tchar = s[iend-1];
	// cout << "Previous char is '" << tchar << "'" << endl;
	// See if separated
	if ( (('a' <= tchar) && (tchar <= 'z')) || 
	     (('A' <= tchar) && (tchar <= 'Z')) || 
	     (('0' <= tchar) && (tchar <= '9')) || 
	     ( tchar == '_') ) 
	  {
	    res += oldstr;
	    iend += oldstr.length();
	    // cout << "not beginning of a word." << endl;
	    continue;
	  }
      }
      // cout << "could be beginning of a word." << endl;
      iend += oldstr.length();
      if ( iend < s.length() ) {
	tchar = s[iend];
	// cout << "Next char is '" << tchar << "'" << endl;
	// See if separated
	if ( (('a' <= tchar) && (tchar <= 'z')) || 
	     (('A' <= tchar) && (tchar <= 'Z')) || 
	     (('0' <= tchar) && (tchar <= '9')) || 
	     ( tchar == '_') ) {
	  res += oldstr;
	  // cout << "not end of a word." << endl;
	  continue;
	}
      }
      // cout << "is a word." << endl;
      // Is separated
      res += newstr;
    }
  }
  return res;
}

// Find the closing paren of a string
size_t findClosingParen(const std::string& s, size_t b)
{
  int count = 1;
  for (size_t i=b; i<s.length(); ++i)
    {
      if (s[i]==')') 
	{
	  count--;
	  if (!count) return i;
	}
      else if (s[i]=='(') count++;
    }
  return s.length();
}

/**
 * Return the number of characters that have been read
 * from the stream
 *
 * @param sstr the stringstream.
 *
 * @return the number of characters read or -1 if hit eof, which indicates that a valid value was not read.
 */

int getReadChars(std::stringstream& sstr) 
{
  
  int count;
#if defined(__DECCXX)
  streampos spos = sstr.tellg();
  count = (spos.offset() == EOF ? -1 : spos.offset() );
#elif defined(__GNUC__) || defined(__EDG) || defined(__KCC)
  std::streampos spos = sstr.tellg();
  count = ( (int) spos == EOF ? -1 : (int) spos );
#elif defined(__BCPLUSPLUS__)
  count = sstr.pcount();
#else
  count = sstr.gcount();    //  This does not work on AIX or DEC
#endif
  // cerr << "getReadChars returning " << count << endl;
  return count;
}

