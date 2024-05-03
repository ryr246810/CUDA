//--------------------------------------------------------------------
//
// File:	TxHierAttribSet.cpp
//
// Purpose:	Implementation of a general holder of a set of attributes.
//
// Copyright (c) 1998 Tech-X Corporation
//
// All rights reserved.
//
// Version:	$Id: TxHierAttribSet.cpp 65 2006-09-25 23:14:25Z yew $
//
//--------------------------------------------------------------------

// Unix includes

// Local includes
#include <TxHierAttribSet.h>

//
// Constructors and destructor
//

TxHierAttribSet::TxHierAttribSet()		// Protected constructor
{
  publicReshape = false;
  setType(getClassName());
  setup();
}

TxHierAttribSet::TxHierAttribSet(std::string s) 	// Public constructor
{
  setObjectName(s);
  publicReshape = true;
  setType(getClassName());
  setup();
}

TxHierAttribSet::TxHierAttribSet(const TxHierAttribSet& txp) : TxAttributeSet(txp)
{
  
  // Set local stuff
  setType(txp.getType());
  
  // Add all data of object to be copied
  if (txp.attribIndicesOwner)
    {
      attribIndicesOwner = true;
      attribIndices = new std::map< std::string, int,std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator mapIter;
      for (mapIter=txp.attribIndices->begin(); mapIter!=txp.attribIndices->end(); mapIter++) attribIndices->insert(*mapIter);
    }
  else
    {
      attribIndicesOwner = false;
      attribIndices = txp.attribIndices;
    }
  for (size_t i=0; i<txp.attribs.size(); i++) 
    {
      attribs.push_back(new TxHierAttribSet(*(txp.attribs[i])) );
    }
  setToFirstAttrib();
  
}

TxHierAttribSet::~TxHierAttribSet() 
{
  if (attribIndicesOwner) delete attribIndices;
  for (size_t i=0; i<attribs.size(); i++) 
    {
      delete attribs[i];
    }
}

TxHierAttribSet& TxHierAttribSet::operator=(const TxHierAttribSet& txp) 
{
  // cerr << "TxHierAttribSet::operator=: entered." << std::endl;
  if (this==&txp) return *this;
  
  // Set base class information
  TxAttributeSet::operator=(txp);
  
  // Add all data of object to be copied
  if (publicReshape) 
    {
      if (attribIndicesOwner) delete attribIndices;
      if (txp.attribIndicesOwner)
	{
	  attribIndicesOwner = true;
	  attribIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator mapIter;
	  for (mapIter=txp.attribIndices->begin(); mapIter!=txp.attribIndices->end(); mapIter++)
	    {
	      attribIndices->insert(*mapIter);
	    }
    }
    else
      {
	attribIndicesOwner = false;
	attribIndices = txp.attribIndices;
      }
      for (size_t i=0; i<txp.attribs.size(); i++) 
	{
	  attribs.push_back(new TxHierAttribSet(*(txp.attribs[i])) );
	}
      setToFirstAttrib();
      return *this;
    }
  
  // Options
  txp.setToFirstAttrib();
  int i;
  for (i=0; i<txp.getNumAttribs(); i++) 
    {
      std::pair<std::string, TxHierAttribSet> attribPair = txp.getCurrentAttribAndBump();
      setAttrib(attribPair.second);
    }
  
  // Done
  return *this;
}

//
// Attriberties accessors
//

bool TxHierAttribSet::hasAttrib(std::string attribName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = attribIndices->find(attribName);
  if ( iter == attribIndices->end() ) return false;
  return true;
}

std::vector<std::string> TxHierAttribSet::getNamesOfType( std::string tp) const 
{
  std::vector<std::string> res;
  for (size_t i=0; i<attribs.size(); ++i) 
    {
      if (attribs[i]->getType() == tp) res.push_back(attribs[i]->getObjectName());
    }
  return res;
}

// Remove an attribute of a given name
void TxHierAttribSet::removeAttrib(const std::string &attribName) 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = attribIndices->find(attribName);
  if (iter == attribIndices->end()) return;
  const int idx = iter->second;
  delete attribs[idx];
  attribs.erase(attribs.begin()+idx);
  attribIndices->erase(iter);
  for (iter=attribIndices->begin(); iter!=attribIndices->end(); iter++) 
    {
      if (idx<iter->second) iter->second--;
    }
}

void TxHierAttribSet::setAttrib(const TxHierAttribSet& value) 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = attribIndices->find(value.getObjectName());
  if ( iter != attribIndices->end() ) 
    {
      delete attribs[iter->second];
      attribs[iter->second] = new TxHierAttribSet(value);
    }
  else std::cout << "MISSING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   " << getObjectName() << "  " << value.getObjectName() << '\n';
}

TxHierAttribSet TxHierAttribSet::getAttrib(std::string attribName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = attribIndices->find(attribName);
  if ( iter != attribIndices->end() ) return TxHierAttribSet( *(attribs[iter->second]) );
  std::string msg("No such attribute '");
  msg += attribName;
  msg += "'";
  throw TxDebugExcept(msg);
}

int TxHierAttribSet::getAttribIndex(std::string attribName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = attribIndices->find(attribName);
  if ( iter != attribIndices->end() ) return iter->second;
  else return getNumAttribs();
}

std::pair<std::string, TxHierAttribSet>TxHierAttribSet::getCurrentAttribAndBump() const 
{
  // JRC fix?
  // return std::pair<std::string, TxHierAttribSet>(paramIter->first,
  // TxHierAttribSet( *(attribs[attribIter->second])) );
  std::pair<std::string, TxHierAttribSet> p(attribIter->first,TxHierAttribSet( *(attribs[attribIter->second])) );
  ++attribIter;
  if (attribIter == attribIndices->end()) attribIter = attribIndices->begin();
  return p;
}

void TxHierAttribSet::emptyNoChk() 
{
  if (attribIndicesOwner) delete attribIndices;
  for (size_t idx = attribs.size(); 0<idx; delete attribs[--idx]);
  attribs.clear();
  setup();
  TxAttributeSet::emptyNoChk();
}

//
// I/O
//
void TxHierAttribSet::writeBody(std::ostream& ostr, bool dump,size_t indentation) const 
{
  
  TxAttributeSet::writeBody(ostr, dump, indentation);
  ++indentation;
  
  std::map<std::string, int, std::less<std::string> >::iterator iter;
  
  // Output params
  if (attribIndices->size())
    {
      if (dump) indent(ostr, indentation), ostr << "<attribs>" << std::endl;
      for (iter=attribIndices->begin(); iter!=attribIndices->end(); iter++)
	{
	  attribs[iter->second]->writeHead(ostr, indentation);
	  attribs[iter->second]->writeBody(ostr, dump, indentation);
	  attribs[iter->second]->writeTail(ostr, indentation);
	}
      if (dump) indent(ostr, indentation), ostr << "</attribs>" << std::endl;
    }
  
}

std::string TxHierAttribSet::readType(std::istream& istr, const std::string& tp) 
{
  ////////////////////////////// Try base class read////////////////////////////
  if ( !TxAttributeSet::readType(istr, tp).length() ) return "";
  
  //////////////////////////// Determine type. If attribs, then quit on endattribs////////////////////////
  std::string classid;
  std::string quitstr;
  if ( tp == "attribs" ){
    ///////////////// Keep reading until the line is </attribs>//////////////////////
    while ( ! istr.eof() ){
      std::string line = TxAttributeSet::getAnyPlatformLine(istr, commentChar);
      
      ///////////////////////// Determine whether noting a type/////////////////////
      if (line[0]=='<')  {
	int endIndex = line.find('>');  if (endIndex == -1) return line;
	int startIndex = line.find_first_not_of(" \t", 1);   if (startIndex == endIndex) return line;
	if ( line[startIndex] == '/' ) return line;
	endIndex = line.find_last_not_of(" \t", endIndex) + 1;
	std::string newtype = line.substr(startIndex, endIndex-startIndex); //if( newtype == "/attribs" ) return "";  // modified by wy 2010.08.12
	std::string retline = readType(istr, newtype);
	if (retline.length()) return retline;
      }
      else{
	//////////////////// Not obeying the format/////////////////
	return line;
      }
      return "";
    }
    }
  else {
    //////////////////// Try reading the one TxAttributeSet then return///////////////////////////
    int whiteIndex = tp.find_first_of(" \t", 0); // the whiteIndex is between the theType string and nameid string, use to split these two strings
    if ( whiteIndex == -1 ) whiteIndex = tp.length();
    std::string theType = tp.substr(0, whiteIndex);
    int nameIndex = tp.find_first_not_of(" \t", whiteIndex);
    std::string nameid;
    if ( nameIndex == -1 ) {
      nameid = "";
    }
    else {
      whiteIndex = tp.find_first_of(" \t", nameIndex);
      if ( whiteIndex == -1 ) whiteIndex = tp.length();
      nameid = tp.substr(nameIndex, whiteIndex - nameIndex);
    }
    TxHierAttribSet newAttrib(nameid);
    newAttrib.setType(theType);

    ///////////////If a new attribute set is found, the operation ">>" is recursively called.///////////////
    istr >> newAttrib; 
    if (hasAttrib(nameid)) setAttrib(newAttrib);
    else if (publicReshape) appendAttribNoChk(newAttrib);
    return "";
  }
  return "";
}


//
// Appending new data
//

bool TxHierAttribSet::appendAttribNoChk(const TxHierAttribSet& value) 
{
  std::string attribName = value.getObjectName();
  attribs.push_back(new TxHierAttribSet(value));
  bool res = attribIndices->insert(std::pair<const std::string, int>(attribName, attribs.size()-1)).second;
  if (!res) 
    {	// Cannot add second key of same type
      delete attribs[attribs.size()-1];
      attribs.pop_back();
      return false;
    }
  return true;
}

//
// Initializing
//
void TxHierAttribSet::setup()
{
  attribIndices = new std::map< std::string, int, std::less<std::string> >;
  attribIndicesOwner = true;
  attribIter = attribIndices->begin();
}

std::string TxHierAttribSet::getClassName() const 
{
  return ("TxHierAttribSet");
}

std::string TxHierAttribSet::getBaseClass() const 
{
  return ("TxAttributeSet");
}
//
// Static data
//
//const std::string TxHierAttribSet::className("TxHierAttribSet");
//const std::string TxHierAttribSet::baseClassName("TxAttributeSet");

