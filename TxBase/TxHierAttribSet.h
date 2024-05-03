//--------------------------------------------------------------------
//
// File:	TxHierAttribSet.h
//
// Purpose:	Interface for a hierarchical holder of a set of attributes.
//		That is, TxHierAttribSet can include themselves.
//
// Copyright (c) 1998 Tech-X Corporation
//
// All rights reserved.
//
// Version:	$Id: TxHierAttribSet.h 58 2006-09-18 17:25:05Z yew $
//
//--------------------------------------------------------------------

#ifndef TX_HIER_ATTRIB_SET_H
#define TX_HIER_ATTRIB_SET_H
#include <TxBase.h>
#ifdef _WIN32
// #pragma warning ( disable: 4786)
// innocent pragma to keep the VC++ compiler quiet about the too long names in STL headers.
#endif

// Local includes
#include <TxAttributeSet.h>

#define TX_ATTRIB_SET_TYPE	10

/**
 * A hierarchical attribute set contains all that an attribute set
 * does, plus more: it can itself contain hierarchical attribute sets.
 */
class TXBASE_API TxHierAttribSet : public TxAttributeSet 
{
 public:
/**
 * Default constructor.
 */
  TxHierAttribSet();
  
/**
 * Construct with a given name - not reshapable
 *
 * @param s the name of the new TxHierAttrib.
 */
  TxHierAttribSet(std::string s);

/**
 * Constructor from a TxHierAttribSet.
 *
 * @param tha the TxHierAttribSet to copy.
 */
  TxHierAttribSet(const TxHierAttribSet& tha);
/**
 * Constructor from a TxAttributeSet.
 *
 * @param tp the ThAttributeSet to copy.
 */
  TxHierAttribSet(const TxAttributeSet& tp) : TxAttributeSet(tp) { setup(); }
    
/**
 * Destructor
 */
  virtual ~TxHierAttribSet();

/**
 * Assign from a TxHierAttribSet.
 *
 * @param tha the TxHierAttribSet to assign from.
 */
  TxHierAttribSet& operator=(const TxHierAttribSet& tha);
/**
 * Assign from a TxAttributeSet.
 *
 * @param tp the TxAttributeSet to assign from.
 */
  TxHierAttribSet& operator=(const TxAttributeSet& tp) 
    {
      TxAttributeSet::operator=(tp);
      return *this;
    }
    
// Accessors for class
/**
 * get the class name
 *
 * @return the name of this class
 */
  virtual std::string getClassName() const; 

/**
 * get the name of the base class, a class containing the same
 * types of data, but perhaps different length lists or
 * different names
 *
 * @return the data names
 */
  virtual std::string getBaseClass() const;

//
// Accessors for attribs (int's)
//

/**
 * get whether one has an attribute
 *
 * @param attribName the name of the attribute to get
 *
 * @return whether this attribute is present
 */
  bool hasAttrib(std::string attribName) const;

/**
 * get list of all attributes of a certain type
 *
 * @param type the type of the attribute to get
 *
 * @return list of
 */
  std::vector<std::string> getNamesOfType(std::string type) const;

/**
 * Set the value of an attribute
 *
 * @param value the new value for a contained attribute of the same name
 */
  virtual void setAttrib(const TxHierAttribSet& value); // Looks up by name

/**
 * Get an attribute of a given name
 *
 * @param attribName the name of the attribute
 *
 * @return the attribute of this name
 */
  TxHierAttribSet getAttrib(std::string attribName) const;

/**
 * Remove an attribute of a given name
 *
 * @param attribName the name of the attribute
 */
  void removeAttrib (const std::string &attribName);
    
/**
 * Get the index of an attribute
 *
 * @param attribName the name of the attribute
 *
 * @return the index of this attribute
 */
  int getAttribIndex(std::string attribName) const;

/**
 * Append a new attribute
 *
 * @param value the new attribute - lookups will key off of its name
 *
 * @return whether the appending was successful
 */
  bool appendAttrib(const TxHierAttribSet& value)
    {
      if (publicReshape) return appendAttribNoChk(value);
      else return false;
    }
  
/**
 * Get the number of attributes
 *
 * @return the number of attributes
 */
  int getNumAttribs() const { return attribs.size();}

/**
 * Set the internal iterator to the first attribute
 */
  void setToFirstAttrib() const { attribIter = attribIndices->begin();}

/**
 * get name of and attribute currently pointed to by the internal
 * iterator and increment the iterator.
 */
  std::pair<std::string, TxHierAttribSet> getCurrentAttribAndBump() const;

/**
 * write the text representation
 *
 * @param ostr the stream to write to
 * @param dump whether to dump full info; defaults to true
 * @param indentation number of times to indent
 */
  virtual void writeBody(std::ostream& ostr, bool dump=true, size_t indentation=0) const;

/**
 * Read one or more data members of a given type
 *
 * @param istr the input stream to read
 * @param type the type to be read
 *
 * @return whether read
 */
  virtual std::string readType(std::istream& istr, const std::string& type);

 protected:

  /**
   * Remove all options, parameters, and strings without checking
   * whether okay.
   */
  virtual void emptyNoChk();
    
/**
 * Protected appends - publicReshape not checked.
 */
  bool appendAttribNoChk(const TxHierAttribSet& value);

// Data
/**
 * Lookup of the index of an attribute of a given name.
 * The index is used in the vector to get the value.
 */
  std::map< std::string, int, std::less<std::string> >* attribIndices;

/**
 * Whether the attribIndices are owned by this object and so should be
 * destroyed upon object destruction.
 */
  bool attribIndicesOwner;

/**
 * The attributes vector.
 */
  std::vector<TxHierAttribSet*> attribs;	// Metrowerks: this a pointer

/**
 * Internal iterator that tells which attribute (in the order of attribIndices)
 * is the current attribute.
 */
  mutable std::map< std::string, int, std::less<std::string> >:: iterator attribIter;
  
 private:
 /**
 * Utility function of construction
 */
  void setup();

/** class name */
  static const std::string className;

/** base class name */
  static const std::string baseClassName;

};

// I/O
inline std::ostream& operator<<(std::ostream& ostr, const TxHierAttribSet& txp) 
{
  txp.write(ostr);
  return ostr;
}

inline std::istream& operator>>(std::istream& istr, TxHierAttribSet& txp) 
{
  txp.read(istr);
  return istr;
}

#endif	// TX_HIER_ATTRIB_SET_H

