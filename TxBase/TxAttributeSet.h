//--------------------------------------------------------------------
//
// File:    TxAttributeSet.h
//
// Purpose: Interface for a general holder of a set of attributes.
//
// Version: $Id: TxAttributeSet.h 142 2008-01-24 19:46:53Z yew $
//
// Copyright (c) 1999 by Tech-X Corporation.  All rights reserved.
//
//--------------------------------------------------------------------

#ifndef TX_ATTRIBUTE_SET_H
#define TX_ATTRIBUTE_SET_H
#include <TxBase.h>
// innocent pragma to keep the VC++ compiler quiet about the too long names in STL headers.
#ifdef _WIN32
// #pragma warning (disable: 4786)
// #pragma warning (disable: 4290)
#endif

// system includes
#include <map>
#include <vector>

// txbase includes
#include <TxDebugExcept.h>
#include <TxStrManip.h>

#define TX_OPTION_TYPE	0
#define TX_PARAM_TYPE	1
#define TX_STRING_TYPE	2
#define TX_OPTVEC_TYPE	3
#define TX_PRMVEC_TYPE	4
#define TX_STRVEC_TYPE	5

/**
 *  TxAttributeSet is a class to encapsalate a set of attributes.
 */
class TXBASE_API TxAttributeSet 
{

 public:

/**
 * Construct an attribute set that is reshapable of given name
 *
 * @param s the name of the new TxAttributeSet.
 */
  TxAttributeSet(std::string s);

/**
 * Copy constructor makes an exact duplicate.
 */
  TxAttributeSet(const TxAttributeSet&);

/**
 * Destructor
 */
  virtual ~TxAttributeSet();

/**
 * Assignment set identically if publicReshape, otherwise elements
 * of same names set.
 */
  TxAttributeSet& operator=(const TxAttributeSet&);
    
/**
 * Set this object from an attribute set
 *
 * @param txs the TxAttributeSet to set from
 */
  virtual void setFromAttribSet(const TxAttributeSet& txs) { operator=(txs); }

/**
 * Get an attribute set description of this object.  This function
 * is defined to allow creation, given that the base class has
 * this method as pure virtual.  However, it is never used.
 *
 * @return the attribute set description
 */
  virtual TxAttributeSet getAttribSetDesc() const { return *this; }

// Accessors for class, name

/**
 * Return the class name.
 *
 * @return class name
 */
  virtual std::string getClassName() const;

/**
 * GetBaseClass returns name of base class that has same types of data
 * though possibly data of different names.
 *
 * @return base class name
 */
  virtual std::string getBaseClass() const;

/**
 * Set the name of this object.
 *
 * @param nm the new name
 */
  void setObjectName(const std::string& nm) { objName = nm;}

/**
 * Return the name of this object.
 *
 * @return the name of the object
 */
  std::string getObjectName() const { return objName;}

/**
 * Set the type of this object.  The type is used for creating
 * derived types that have the same data structure.  It stores
 * the character name of what should be created from this.
 *
 * @param tp the new type
 */
  void setType(const std::string& tp) { type = tp;}

/**
 * get the type of this object
 *
 * @return the type of the object
 */
  std::string getType() const { return type;}

//
// Accessors for options (int's)
//

/**
 * Return whether has an option of this name
 *
 * @return whether this has this option
 */
  bool hasOption(const std::string& optionName) const;

/**
 * Set the value of the option of this name
 *
 * @param optionName the name of this option
 * @param value the new value of this option
 */
  virtual void setOption(const std::string& optionName, int value);

/**
 * Get the value of the option of this name
 *
 * @param optionName the name of the option under consideration
 * @return the value of the option of this name
 */
  int getOption(const std::string& optionName) const ;

/**
 * Get the index of the option of this name
 *
 * @param optionName the name of the option
 * @return the index of the option of this name
 */
  int getOptionIndex(const std::string& optionName) const;

/**
 * Append an option of this name and the given value
 *
 * @param optionName the name of the option to be appended
 * @param value the value of the option to be appended
 *
 * @return whether option successfully appended
 */
  bool appendOption(const std::string& optionName, int value);

/**
 * get the number of options
 *
 * @return the number of options
 */
  int getNumOptions() const { return options.size(); }

/**
 * Set to the first option for incrementing
 */
  void setToFirstOption() const { optionIter = optionIndices->begin();}
    
/**
 * get name of and option currently pointed to, increment option
 *
 * @return pair corresponding to first option
 */
  std::pair<std::string, int> getCurrentOptionAndBump() const;

//
// Accessors for params (int's)
//

/**
 * Determine whether contains a parameter of a given name
 *
 * @param paramName the name to test for
 *
 * @return whether a parameter of that name is present
 */
  bool hasParam(const std::string& paramName) const;

/**
 * Set the value of the parameter of this name
 *
 * @param paramName the name of this parameter
 * @param value the value of this parameter
 */
  virtual void setParam(const std::string& paramName, double value);

/**
 * Get the value of the parameter of this name
 *
 * @param paramName the name of this parameter
 *
 * @return the value of this parameter
 */
  double getParam(const std::string& paramName) const ;

/**
 * Get the index of a parameter of a certain name
 *
 * @param paramName the name of the parameter
 */
  int getParamIndex(const std::string& paramName) const;

/**
 * Append another parameter
 *
 * @param paramName the name of this parameter
 * @param value the new value for the parameter of this name
 *
 * @return whether parameter successfully appended
 */
  bool appendParam(const std::string& paramName, double value);

/**
 * Get the number of parameters
 *
 * @return the number of parameters
 */
  int getNumParams() const { return params.size();}

/**
 * Set internal iterator to first parameter (alphabetically)
 */
  void setToFirstParam() const { paramIter = paramIndices->begin();}

/**
 * get name of and param currently pointed to, increment param
 */
  std::pair<std::string, double> getCurrentParamAndBump() const;

//
// Accessors for strings
//

/**
 * Determine whether has a std::string of a given name
 *
 * @param stringName the name under consideration
 *
 * @return whether has a std::string of this name
 */
  bool hasString(const std::string& stringName) const;

/**
 * Set the value of a std::string
 *
 * @param stringName the name of the std::string
 * @param value the new value of the std::string
 */
  virtual void setString(const std::string& stringName, const std::string& value);

/**
 * Get a std::string of a given name
 *
 * @param stringName the name of the std::string
 *
 * @return the value of the std::string
 */
  std::string getString(const std::string& stringName) const ;

/**
 * Get the index of a std::string
 *
 * @param stringName the name of the std::string
 *
 * @return the value of the std::string of this name
 */
  int getStringIndex(const std::string& stringName) const;

/**
 * Append a std::string
 *
 * @param stringName the name of the std::string to be appended
 * @param value the value of the std::string to be appended
 *
 * @return whether successfully appended
 */
  bool appendString(const std::string& stringName, const std::string& value);

/**
 * Get the number of strings
 *
 * @return the number of strings
 */
  int getNumStrings() const { return strings.size();}

/**
 * Set the internal iterator to the first (alphabetically by name)
 * std::string.
 */
  void setToFirstString() const { stringIter = stringIndices->begin();}

/**
 * get name of and std::string currently pointed to, then increment
 * the internal pointer.
 */
  std::pair<std::string, std::string> getCurrentStringAndBump() const;

//
// Accessors for optVecs (int's)
//

/**
 * Return whether has an optVec of this name
 *
 * @return whether this has this optVec
 */
  bool hasOptVec(const std::string& optVecName) const;

/**
 * Set the value of the optVec of this name
 *
 * @param optVecName the name of this optVec
 * @param value the new value of this optVec
 */
  virtual void setOptVec(const std::string& optVecName, const std::vector<int>& value);

/**
 * Get the value of the optVec of this name
 *
 * @param optVecName the name of the optVec under consideration
 *
 * @return the value of the optVec of this name
 */
  std::vector<int> getOptVec(const std::string& optVecName) const ;

/**
 * Get the index of the optVec of this name
 *
 * @param optVecName the name of the optVec
 *
 * @return the index of the optVec of this name
 */
  int getOptVecIndex(const std::string& optVecName) const;

/**
 * Append an optVec of this name and the given value
 *
 * @param optVecName the name of the optVec to be appended
 * @param value the value of the optVec to be appended
 *
 * @return whether optVec successfully appended
 */
  bool appendOptVec(const std::string& optVecName, const std::vector<int>& value);

/**
 * get the number of optVecs
 *
 * @return the number of optVecs
 */
  int getNumOptVecs() const { return optVecs.size(); }

/**
 * Set to the first optVec for incrementing
 */
  void setToFirstOptVec() const { optVecIter = optVecIndices->begin();}

/**
 * get name of and optVec currently pointed to, increment optVec
 *
 * @return pair corresponding to first optVec
 */
  std::pair< std::string, std::vector<int> > getCurrentOptVecAndBump() const;

//
// Accessors for prmVecs (double's)
//

/**
 * Return whether has an prmVec of this name
 *
 * @return whether this has this prmVec
 */
  bool hasPrmVec(const std::string& prmVecName) const;

/**
 * Set the value of the prmVec of this name
 *
 * @param prmVecName the name of this prmVec
 * @param value the new value of this prmVec
 */
  virtual void setPrmVec(const std::string& prmVecName, const std::vector<double>& value);

/**
 * Get the value of the prmVec of this name
 *
 * @param prmVecName the name of the prmVec under consideration
 *
 * @return the value of the prmVec of this name
 */
  std::vector<double> getPrmVec(const std::string& prmVecName) const ;

/**
 * Get the index of the prmVec of this name
 *
 * @param prmVecName the name of the prmVec
 *
 * @return the index of the prmVec of this name
 */
  int getPrmVecIndex(const std::string& prmVecName) const;

/**
 * Append an prmVec of this name and the given value
 *
 * @param prmVecName the name of the prmVec to be appended
 * @param value the value of the prmVec to be appended
 *
 * @return whether prmVec successfully appended
 */
  bool appendPrmVec(const std::string& prmVecName, const std::vector<double>& value);

/**
 * get the number of prmVecs
 *
 * @return the number of prmVecs
 */
  int getNumPrmVecs() const { return prmVecs.size(); }

/**
 * Set to the first prmVec for incrementing
 */
  void setToFirstPrmVec() const { prmVecIter = prmVecIndices->begin();}

/**
 * get name of and prmVec currently pointed to, increment prmVec
 *
 * @return pair corresponding to first prmVec
 */
  std::pair< std::string, std::vector<double> > getCurrentPrmVecAndBump() const;

//
// Accessors for strVecs (string's)
//

/**
 * Return whether has an strVec of this name
 *
 * @return whether this has this strVec
 */
  bool hasStrVec(const std::string& strVecName) const;

/**
 * Set the value of the strVec of this name
 *
 * @param strVecName the name of this strVec
 * @param value the new value of this strVec
 */
  virtual void setStrVec(const std::string& strVecName, const std::vector<std::string>& value);

/**
 * Get the value of the strVec of this name
 *
 * @param strVecName the name of the strVec under consideration
 *
 * @return the value of the strVec of this name
 */
  std::vector<std::string> getStrVec(const std::string& strVecName) const ;

/**
 * Get the index of the strVec of this name
 *
 * @param strVecName the name of the strVec
 *
 * @return the index of the strVec of this name
 */
  int getStrVecIndex(const std::string& strVecName) const;

/**
 * Append an strVec of this name and the given value
 *
 * @param strVecName the name of the strVec to be appended
 * @param value the value of the strVec to be appended
 *
 * @return whether strVec successfully appended
 */
  bool appendStrVec(const std::string& strVecName, const std::vector<std::string>& value);

/**
 * get the number of strVecs
 *
 * @return the number of strVecs
 */
  int getNumStrVecs() const { return strVecs.size(); }

/**
 * Set to the first strVec for incrementing
 */
  void setToFirstStrVec() const { strVecIter = strVecIndices->begin();}

/**
 * get name of and strVec currently pointed to, increment strVec
 *
 * @return pair corresponding to first strVec
 */
  std::pair<std::string, std::vector<std::string> > getCurrentStrVecAndBump() const;


/**
 * Return the names of this object.
 * The names will be the string names of each of the types
 *
 * @return the vector of names of this object
 */
  std::vector<std::string> getNames() const { return names;}

/**
 * Return the type names of this object.
 * The type names will be the names of each of the types, ie
 * options for all the Option types
 * strings for all the String types
 * params for all the Param types, etc
 *
 * @return the vector of types of this object
 */
  std::vector<std::string> getTypes() { return types; }

/**
 * Return the integer type names of this object.
 * The integer type names will be the #defines of each of the types, ie
 * TX_OPTION_TYPE for all the options types, etc
 *
 * @return the vector of types of this object
 */
  std::vector<int> getIntTypes();

//
// End of accessors
//

/**
 * clear out all data if allowed
 */
  void empty() { if (publicReshape) emptyNoChk(); }

/**
 * Boolean operator - for sorting in alphabetical order
 *
 * @param a returns whether the object's name is earlier than the other
 */
  virtual bool operator<(const TxAttributeSet& a) const { return getObjectName() < a.getObjectName(); }

/**
 * Boolean operator for names
 *
 * @param a returns whether the two objects have the same name
 */
  virtual bool operator==(const TxAttributeSet& a) const { return getObjectName() == a.getObjectName(); }

/**
 * Boolean inequality for names
 *
 * @param a returns whether the two objects have the same name
 */
  virtual bool operator!=(const TxAttributeSet& a) const { return ! operator==(a); }


/**
 * Write out the attributes text representation
 *
 * @param ostr the stream to write to
 * @param dump whether to dump full info; defaults to true
 */
  //virtual void write(std::ostream& ostr, bool dump=true) const;
  virtual void write(std::ostream& ostr, bool dump = true, size_t indentation = 1) const;
/**
 * read from the istream
 *
 * @param istr the stream to read from
 */
  virtual std::string read(std::istream& istr);


 protected:
/**
 * Write out the head of the attributes text representation
 *
 * @param ostr the stream to write to
 * @param indentation number of times to indent
 */
  virtual void writeHead(std::ostream& ostr, size_t indentation=0) const;

/**
 * Write out the body of the attributes text representation
 *
 * @param ostr the stream to write to
 * @param dump whether to dump full info; defaults to true
 * @param indentation number of times to indent
 */
  virtual void writeBody(std::ostream& ostr, bool dump=true, size_t indentation=0) const;

/**
 * Write out the end of the attributes text representation
 *
 * @param ostr the stream to write to
 * @param indentation number of times to indent
 */
  virtual void writeTail(std::ostream& ostr, size_t indentation=0) const;

// Why is this static?  JRC will remove.  Static prevents re-entrant.
    // static void indent(std::ostream& out);
  void indent(std::ostream& out, size_t indentation) const;

    // static int indentation;
    // mutable int indentation;

/**
 * Read one or more data members of a given type
 *
 * @param istr the stream to read from
 * @param type the type to be read
 *
 * @return whether read
 */
  virtual std::string readType(std::istream& istr, const std::string& type);

/**
 * Read a data member of an unknown type from a given line
 *
 * @param line the line to be parsed
 *
 * @return whether read
 */
  virtual bool parseLine(const std::string& line);

/**
 * read a string vector from a stream, then puts it into a vector of strings.
 *
 * @param sstr the input stream.
 * @param vals the vector of strings parsed.
 *
 */
  static void readQuotedStringVecs(std::istream& sstr,  std::vector<std::string>& vals);

/**
 * Trim white space from a line up to comment character
 *
 * @param line the line to be trimmed
 *
 * @return the trimmed line
 */
  static std::string trimLine(const std::string& line, char commentchar);

/**
 * Utility function for getLine that works on mac/linux/dos.
 *
 * @param istr the stream to get line from
 */
  static std::string getAnyPlatformLine(std::istream& istr, char commentchar);

/**
 * Read a line of data.  A line ends at a newline, unless it contains
 * a left bracket, in which case it continues to add lines until the
 * bracket is closed.
 *
 * @param istr the stream to read from
 *
 * @return the trimmed line
 */
  virtual std::string getLine(std::istream& istr) ;

/**
 * Get the number of a given char in a string.
 *
 * @param s the string to parse
 * @param c the char to look for
 *
 * @return the number of a given char
 */
  virtual size_t getNumChars(const std::string& s, char c)
    {
      size_t num=0;
      for (size_t i=0; i<s.length(); ++i) if (s[i] == c) num++;
      return num;
    }

/**
 * Get name and value strings from a line
 *
 * @param line the line to be parsed
 *
 * @return the pair, name & value std::string
 */
  virtual std::pair<std::string, std::string> getNameAndValueStr(std::string line);

/**
 * Construction without a name is protected.
 * Generally one cannot append.
 */
  TxAttributeSet();

/**
 * Protected appends - publicReshape not checked.
 */

/**
 * Append an option without checking whether okay
 *
 * @param optionName the name of the option to append
 * @param value the value of the new option
 */
  bool appendOptionNoChk(const std::string& optionName, int value);

/**
 * Append a parameter without checking whether okay
 *
 * @param paramName the name of the parameter to append
 * @param value the value of the new parameter
 */
  bool appendParamNoChk(const std::string& paramName, double value);

/**
 * Append a std::string without checking whether okay
 *
 * @param stringName the name of the std::string to append
 * @param value the value of the new std::string
 */
  bool appendStringNoChk(const std::string& stringName, const std::string& value);

/**
 * Append an optVec without checking whether okay
 *
 * @param optVecName the name of the optVec to append
 * @param value the value of the new optVec
 */
  bool appendOptVecNoChk(const std::string& optVecName, const std::vector<int>& value);

/**
 * Append an prmVec without checking whether okay
 *
 * @param prmVecName the name of the prmVec to append
 * @param value the value of the new prmVec
 */
  bool appendPrmVecNoChk(const std::string& prmVecName, const std::vector<double>& value);

/**
 * Append an strVec without checking whether okay
 *
 * @param strVecName the name of the strVec to append
 * @param value the value of the new strVec
 */
  bool appendStrVecNoChk(const std::string& strVecName, const std::vector<std::string>& value);

/**
 * Remove all options, parameters, and strings without checking
 * whether okay.
 */
  virtual void emptyNoChk();

/**
 * Whether this can be reshaped: additional data added or removed.
 */
  bool publicReshape;	// Whether public can change data shape (names..)

/**
 * Lookup of the index of an option of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* optionIndices;
/**
 * Whether the optionIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool optionIndicesOwner;	// Whether owns the options
/**
 * The options vector.
 */
  std::vector<int> options;	// Vector of options
/**
 * Internal iterator that tells which option (in the order of optionIndices)
 * is the current option.
 */
  mutable std::map< std::string, int, std::less<std::string> >::iterator optionIter;

// Lookup for params
/**
 * Lookup of the index of a param of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* paramIndices;
/**
 * Whether the paramIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool paramIndicesOwner;	// Whether owns the parameters
/**
 * The params vector.
 */
  std::vector<double> params;	// Vector of parameters
/**
 * Internal iterator that tells which param (in the order of paramIndices)
 * is the current param.
 */
  mutable std::map< std::string, int, std::less<std::string> >::iterator paramIter;

// Lookup for strings
/**
 * Lookup of the index of a string of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* stringIndices;
/**
 * Whether the stringIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool stringIndicesOwner;	// Whether owns the strings
/**
 * The string vector.
 */
  std::vector<std::string> strings;	// Vector of strings
/**
 * Internal iterator that tells which string (in the order of stringIndices)
 * is the current string.
 */
  mutable std::map< std::string, int, std::less<std::string> >::iterator stringIter;

// Lookup for vectors of options
/**
 * Lookup of the index of an option vector of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* optVecIndices;
/**
 * Whether the optVecIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool optVecIndicesOwner;	// Whether owns option vectors
/**
 * The vector of option vectors.
 */
  std::vector< std::vector<int> > optVecs;	// Vector of option vectors
/**
 * Internal iterator that tells which option vector (in the order of
 * optVecIndices) is the current option vector.
 */
  mutable std::map< std::string, int,std::less<std::string> >::iterator optVecIter;

// Lookup for vectors of params
/**
 * Lookup of the index of a param vector of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* prmVecIndices;
/**
 * Whether the prmVecIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool prmVecIndicesOwner;	// Whether owns the param vectors
/**
 * The vector of param vectors.
 */
  std::vector< std::vector<double> > prmVecs;	// Vector of param vectors
/**
 * Internal iterator that tells which param vector (in the order of
 * prmVecIndices) is the current param vector.
 */
  mutable std::map< std::string, int, std::less<std::string> >::iterator prmVecIter;

// Lookup for vectors of strings
/**
 * Lookup of the index of a string vector of a given name.  The index is used
 * in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* strVecIndices;
/**
 * Whether the strVecIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool strVecIndicesOwner;	// Whether owns the string vectors
/**
 * The vector of string vectors.
 */
  std::vector< std::vector< std::string> > strVecs; // Vector of string vectors
/**
 * Internal iterator that tells which string vector (in the order of
 * strVecIndices) is the current string vector.
 */
  mutable std::map< std::string, int, std::less<std::string> >::iterator strVecIter;

/**
 * The character that denotes the beginning of a comment
 */
  char commentChar;

/**
 * The current parse line
 */
  size_t lineNum;

/**
 * The names vector.
 */
  std::vector<std::string> names;	// The names

/**
 * The types vector.
 */
  std::vector<std::string> types;	// The types

 private:
/**
 * Utility function for construction
 */
  void setup();


 private:

/** class name */
//    static const std::string className;

/** base class name */
//    static const std::string baseClassName;

/** Name of this property set */
  std::string objName;

/** Type of this attribute set */
  std::string type;

};

/**
 * output operator of a TxAttributeSet
 */
inline std::ostream& operator<< (std::ostream& ostr, const TxAttributeSet& txp) 
{
  txp.write(ostr);
  return ostr;
}

/**
 * input operator of a TxAttributeSet
 */
inline std::istream& operator>>(std::istream& istr,TxAttributeSet& txp)
{
  txp.read(istr);
  return istr;
}

#endif	// TX_ATTRIBUTE_SET_H
