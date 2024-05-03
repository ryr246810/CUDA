//----------------------------------------------------------------------------
//
// File:    TxAttributeSet.cpp
//
// Purpose: Implementation of a general holder of a set of properties.
//
// Version: Id: TxAttributeSet.cpp  2008-12-26 21:14  China Wang Yue
//
// Copyright (c) 1999-2003 by Tech-X Corporation.  All rights reserved.
//
//----------------------------------------------------------------------------

// Standard includes
#include <cstdlib>

// Local includes
#include <TxAttributeSet.h>
#include <TxDebugExcept.h>
#include <iostream>
using namespace std;

TxAttributeSet::TxAttributeSet() 
{
  // std::cerr << "TxAttributeSet::TxAttributeSet(): entered.\n";
  publicReshape = false;
  type = getClassName();
  setup();
  // std::cerr << "TxAttributeSet::TxAttributeSet(): completed.\n";
}

TxAttributeSet::TxAttributeSet(std::string s) 
{
  objName = s;
  publicReshape = true;
  type = getClassName();
  setup();
}

TxAttributeSet::TxAttributeSet(const TxAttributeSet& txp) 
{
  // Set names and flags
  objName = txp.objName;
  publicReshape = txp.publicReshape;
  type = txp.type;
  commentChar = txp.commentChar;
  
  // Add all data of object to be copied
  if (txp.optionIndicesOwner) 
    {
      optionIndicesOwner = true;
      optionIndices = new std::map< std::string, int,std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator mapIter;
      for (mapIter=txp.optionIndices->begin(); mapIter!=txp.optionIndices->end(); mapIter++) optionIndices->insert(*mapIter);
    }
  else 
    {
      optionIndicesOwner = false;
      optionIndices = txp.optionIndices;
    }
  options = txp.options;
  setToFirstOption();
  
  // Add all data of object to be copied
  if (txp.paramIndicesOwner)
    {
      paramIndicesOwner = true;
      paramIndices = new std::map< std::string, int, std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator mapIter;
      for (mapIter=txp.paramIndices->begin(); mapIter!=txp.paramIndices->end(); mapIter++) paramIndices->insert(*mapIter);
    }
  else 
    {
      paramIndicesOwner = false;
      paramIndices = txp.paramIndices;
    }
  params = txp.params;
  setToFirstParam();
  
  // Add all data of object to be copied
  if (txp.stringIndicesOwner)
    {
      stringIndicesOwner = true;
      stringIndices = new std::map< std::string, int, std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator mapIter;
      for (mapIter=txp.stringIndices->begin(); mapIter!=txp.stringIndices->end(); mapIter++) stringIndices->insert(*mapIter);
    }
  else 
    {
      stringIndicesOwner = false;
      stringIndices = txp.stringIndices;
    }
  strings = txp.strings;
  setToFirstString();
  
  // Add all data of object to be copied
  if (txp.optVecIndicesOwner)
    {
      optVecIndicesOwner = true;
      optVecIndices = new std::map< std::string, int, std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator mapIter;
      for (mapIter=txp.optVecIndices->begin(); mapIter!=txp.optVecIndices->end(); mapIter++) optVecIndices->insert(*mapIter);
    }
  else 
    {
      optVecIndicesOwner = false;
      optVecIndices = txp.optVecIndices;
    }
  optVecs = txp.optVecs ;
  setToFirstOptVec();
  
  // Add all data of object to be copied
  if (txp.prmVecIndicesOwner) 
    {
      prmVecIndicesOwner = true;
      prmVecIndices = new std::map< std::string, int, std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
      for (mapIter=txp.prmVecIndices->begin(); mapIter!=txp.prmVecIndices->end(); ++mapIter) prmVecIndices->insert(*mapIter);
    }
  else 
    {
      prmVecIndicesOwner = false;
      prmVecIndices = txp.prmVecIndices;
    }
  prmVecs = txp.prmVecs ;
  setToFirstPrmVec();
  
  // Add all data of object to be copied
  if (txp.strVecIndicesOwner) 
    {
      strVecIndicesOwner = true;
      strVecIndices = new std::map< std::string, int,  std::less<std::string> >;
      std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
      for (mapIter=txp.strVecIndices->begin(); mapIter!=txp.strVecIndices->end();  mapIter++) strVecIndices->insert(*mapIter);
    }
  else 
    {
      strVecIndicesOwner = false;
      strVecIndices = txp.strVecIndices;
    }
  strVecs = txp.strVecs ;
  setToFirstStrVec();
  
  // assign the names and types vectors
  names = txp.names;
  types = txp.types;
}

TxAttributeSet::~TxAttributeSet() 
{
  if (optionIndicesOwner) delete optionIndices;
  if (paramIndicesOwner) delete paramIndices;
  if (stringIndicesOwner) delete stringIndices;
  if (optVecIndicesOwner) delete optVecIndices;
  if (prmVecIndicesOwner) delete prmVecIndices;
  if (strVecIndicesOwner) delete strVecIndices;
}

TxAttributeSet& TxAttributeSet::operator=(const TxAttributeSet& txp) 
{
  // cerr << "TxAttributeSet::operator=: entered." << std::endl;
  if (this==&txp) return *this;
  
  // Set names and type
  objName = txp.objName;
  type = txp.type;
  publicReshape = txp.publicReshape;
  commentChar = txp.commentChar;
  
  // Set identically to txp if publicReshape is true
  if (publicReshape)
    {
      // Set names and flags
      
      // Add all data of object to be copied
      if (optionIndicesOwner) delete optionIndices;
      if (txp.optionIndicesOwner) 
	{
	  optionIndicesOwner = true;
	  optionIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
	  for (mapIter=txp.optionIndices->begin(); mapIter!=txp.optionIndices->end(); mapIter++) optionIndices->insert(*mapIter);
	}
      else 
	{
	  optionIndicesOwner = false;
	  optionIndices = txp.optionIndices;
	}
      options = txp.options;
      setToFirstOption();
      
      // Add all data of object to be copied
      if (paramIndicesOwner) delete paramIndices;
      if (txp.paramIndicesOwner)
	{
	  paramIndicesOwner = true;
	  paramIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
	  for (mapIter=txp.paramIndices->begin(); mapIter!=txp.paramIndices->end(); mapIter++) paramIndices->insert(*mapIter);
	}
      else 
	{
	  paramIndicesOwner = false;
	  paramIndices = txp.paramIndices;
	}
      params = txp.params;
      setToFirstParam();
      
      // Add all data of object to be copied
      if (stringIndicesOwner) delete stringIndices;
      if (txp.stringIndicesOwner) 
	{
	  stringIndicesOwner = true;
	  stringIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator mapIter;
	  for (mapIter=txp.stringIndices->begin(); mapIter!=txp.stringIndices->end(); mapIter++) stringIndices->insert(*mapIter);
	}
      else 
	{
	  stringIndicesOwner = false;
	  stringIndices = txp.stringIndices;
	}
      strings = txp.strings;
      setToFirstString();
      
      // Add all vector int data
      if (optVecIndicesOwner) delete optVecIndices;
      if (txp.optVecIndicesOwner) 
	{
	  optVecIndicesOwner = true;
	  optVecIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
	  for (mapIter=txp.optVecIndices->begin(); mapIter!=txp.optVecIndices->end(); mapIter++) optVecIndices->insert(*mapIter);
	}
      else 
	{
	  optVecIndicesOwner = false;
	  optVecIndices = txp.optVecIndices;
	}
      optVecs = txp.optVecs;
      setToFirstOptVec();
      
      // Add all vector double data
      if (prmVecIndicesOwner) delete prmVecIndices;
      if (txp.prmVecIndicesOwner)
	{
	  prmVecIndicesOwner = true;
	  prmVecIndices = new std::map< std::string, int,  std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
	  for (mapIter=txp.prmVecIndices->begin(); mapIter!=txp.prmVecIndices->end(); mapIter++) prmVecIndices->insert(*mapIter);
	}
      else 
	{
	  prmVecIndicesOwner = false;
	  prmVecIndices = txp.prmVecIndices;
	}
      prmVecs = txp.prmVecs;
      setToFirstPrmVec();
      
      // Add all vector string data
      if (strVecIndicesOwner) delete strVecIndices;
      if (txp.strVecIndicesOwner)
	{
	  strVecIndicesOwner = true;
	  strVecIndices = new std::map< std::string, int, std::less<std::string> >;
	  std::map< std::string, int, std::less<std::string> >::iterator  mapIter;
	  for (mapIter=txp.strVecIndices->begin(); mapIter!=txp.strVecIndices->end(); mapIter++) strVecIndices->insert(*mapIter);
	}
      else 
	{
	  strVecIndicesOwner = false;
	  strVecIndices = txp.strVecIndices;
	}
      strVecs = txp.strVecs;
      setToFirstStrVec();
      
      // assign the names and types vectors
      names = txp.names;
      types = txp.types;
      
      // For for case  of reshapable
      return *this;
      
    }
  
  // Options
  txp.setToFirstOption();
  int i;
  for (i=0; i<txp.getNumOptions(); i++)
    {
      std::pair<std::string, int> optionPair = txp.getCurrentOptionAndBump();
      setOption(optionPair.first, optionPair.second);
    }
  
  // Params
  txp.setToFirstParam();
  for (i=0; i<txp.getNumParams(); i++) 
    {
      std::pair<std::string, double> paramPair = txp.getCurrentParamAndBump();
      setParam(paramPair.first, paramPair.second);
    }
  
  // Strings
  txp.setToFirstString();
  for (i=0; i<txp.getNumStrings(); i++) 
    {
      std::pair<std::string, std::string> stringPair = txp.getCurrentStringAndBump();
      setString(stringPair.first, stringPair.second);
    }
  
  // Vectors of int's
  txp.setToFirstOptVec();
  for (i=0; i<txp.getNumOptVecs(); i++)
    {
      std::pair<std::string, std::vector<int> > optVecPair = txp.getCurrentOptVecAndBump();
      setOptVec(optVecPair.first, optVecPair.second);
    }
  
  // Vectors of double's
  txp.setToFirstPrmVec();
  for (i=0; i<txp.getNumPrmVecs(); i++)
    {
      std::pair<std::string, std::vector<double> > prmVecPair = txp.getCurrentPrmVecAndBump();
      setPrmVec(prmVecPair.first, prmVecPair.second);
    }
  
  // Vectors of strings's
  txp.setToFirstStrVec();
  for (i=0; i<txp.getNumStrVecs(); i++) 
    {
      std::pair<std::string, std::vector<std::string> > strVecPair = txp.getCurrentStrVecAndBump();
      setStrVec(strVecPair.first, strVecPair.second);
    }

  // assign the names and types vectors
  names = txp.names;
  types = txp.types;
  
  // Done
  return *this;
}

//
// Option accessors
//

//
bool TxAttributeSet::hasOption(const std::string& optionName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  optionIndices->find(optionName);
  if ( iter == optionIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<<"No option named: "<<optionName <<std::endl;
      TxDebugExcept txde("No option named: ");
      txde <<optionName;
      throw txde;
    }
  return true;
}

//
void TxAttributeSet::setOption(const std::string& optionName, int value)
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optionIndices->find(optionName);
  if ( iter != optionIndices->end() ) options[iter->second] = value;
}

//
int TxAttributeSet::getOption(const std::string& optionName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  optionIndices->find(optionName);
  if ( iter == optionIndices->end() ) throw TxDebugExcept(std::string("No option named '") + optionName + "'");
  return options[iter->second];
}

//
int TxAttributeSet::getOptionIndex(const std::string& optionName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optionIndices->find(optionName);
  if ( iter != optionIndices->end() ) return iter->second;
  else return getNumOptions();
}

// Append an option of this name and the given value
bool TxAttributeSet::appendOption(const std::string& optionName, int value) 
{
  if (publicReshape) return appendOptionNoChk(optionName, value);
  else return false;
}

//
std::pair<std::string, int> TxAttributeSet::getCurrentOptionAndBump() const 
{
  std::pair<std::string, int> p(optionIter->first,options[optionIter->second]);
  optionIter++;
  if (optionIter==optionIndices->end()) optionIter = optionIndices->begin();
  return p;
}

//
// Param accessors
//

//
bool TxAttributeSet::hasParam(const std::string& paramName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  paramIndices->find(paramName);
  if ( iter == paramIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<< "No parameter named: "<<paramName<<std::endl;
      TxDebugExcept txde("No parameter named: ");
      txde <<paramName;
      throw txde;
    }
  return true;
}


//
void TxAttributeSet::setParam(const std::string& paramName, double value) 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = paramIndices->find(paramName);
  if ( iter != paramIndices->end() ) params[iter->second] = value;
}

//
double TxAttributeSet::getParam(const std::string& paramName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = paramIndices->find(paramName);
  if ( iter == paramIndices->end() )  throw TxDebugExcept(std::string("No param named '") + paramName + "'");
  return params[iter->second];
}

//
int TxAttributeSet::getParamIndex(const std::string& paramName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  paramIndices->find(paramName);
  if ( iter != paramIndices->end() ) return iter->second;
  else return getNumParams();
}

// Append another parameter
bool TxAttributeSet::appendParam(const std::string& paramName, double value) 
{
  if (publicReshape) return appendParamNoChk(paramName, value);
  else return false;
}




//
std::pair<std::string, double>TxAttributeSet::getCurrentParamAndBump() const 
{
  std::pair<std::string, double> p(paramIter->first, params[paramIter->second]);
  ++paramIter;
  if (paramIter==paramIndices->end()) paramIter = paramIndices->begin();
  return p;
}


//
// String accessors
//

//
bool TxAttributeSet::hasString(const std::string& stringName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = stringIndices->find(stringName);
  if ( iter == stringIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<< "No string named: "<<stringName<<std::endl;
      TxDebugExcept txde("No string named: ");
      txde <<stringName;
      throw txde;
    }
  return true;
}

//
void TxAttributeSet::setString(const std::string& stringName, const std::string& value) 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = stringIndices->find(stringName);
  if ( iter != stringIndices->end() ) strings[iter->second] = value;
}

//
std::string TxAttributeSet::getString(const std::string& stringName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = stringIndices->find(stringName);
  if ( iter == stringIndices->end() ){
    throw TxDebugExcept(std::string("No std::string named '") + stringName + "'");
  }
  return strings[iter->second];
}

//
int TxAttributeSet::getStringIndex(const std::string& stringName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  stringIndices->find(stringName);
  if ( iter != stringIndices->end() ) return iter->second;
  else return getNumStrings();
}

// Append a string
bool TxAttributeSet::appendString(const std::string& stringName, const std::string& value)
{
  if (publicReshape) return appendStringNoChk(stringName, value);
  else return false;
}

//
std::pair<std::string, std::string> TxAttributeSet::getCurrentStringAndBump() const 
{
  std::pair<std::string, std::string> p(stringIter->first, strings[stringIter->second]);
  stringIter++;
  if (stringIter==stringIndices->end()) stringIter = stringIndices->begin();
  return p;
}


//
// OptVec accessors
//

//
bool TxAttributeSet::hasOptVec(const std::string& optVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optVecIndices->find(optVecName);
  if ( iter == optVecIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<<"No Option Vector named: "<<optVecName<<std::endl;
      TxDebugExcept txde("No Option Vector named: ");
      txde <<optVecName;
      throw txde;	  
    }	  
  return true;
}

//
void TxAttributeSet::setOptVec(const std::string& optVecName, const std::vector<int>& value) 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optVecIndices->find(optVecName);
  if ( iter != optVecIndices->end() ) optVecs[iter->second] = value;
}

//
std::vector<int> TxAttributeSet::getOptVec( const std::string& optVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optVecIndices->find(optVecName);
  if ( iter == optVecIndices->end() ) throw TxDebugExcept(std::string("No optVec named '") + optVecName + "'");
  return optVecs[iter->second];
}

//
int TxAttributeSet::getOptVecIndex(const std::string& optVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = optVecIndices->find(optVecName);
  if ( iter != optVecIndices->end() ) return iter->second;
  else return getNumOptVecs();
}

// Append an optVec of this name and the given value
bool TxAttributeSet::appendOptVec(const std::string& optVecName, const std::vector<int>& value) 
{
  if (publicReshape) return appendOptVecNoChk(optVecName, value);
  else return false;
}

//
std::pair< std::string, std::vector<int> > TxAttributeSet::getCurrentOptVecAndBump() const
{
  std::pair<std::string, std::vector<int> > p(optVecIter->first, optVecs[optVecIter->second]);
  optVecIter++;
  if (optVecIter==optVecIndices->end()) optVecIter = optVecIndices->begin();
  return p;
}

//
// PrmVec accessors
//

//
bool TxAttributeSet::hasPrmVec(const std::string& prmVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = prmVecIndices->find(prmVecName);
  if ( iter == prmVecIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<< "No parameter vector named: "<<prmVecName<<std::endl;
      TxDebugExcept txde("No parameter vector named: ");
      txde <<prmVecName;
      throw txde;
    }
  return true;
}

//
void TxAttributeSet::setPrmVec(const std::string& prmVecName, const std::vector<double>& value)
{
  std::map<std::string, int, std::less<std::string> >::iterator iter =  prmVecIndices->find(prmVecName);
  if ( iter != prmVecIndices->end() ) prmVecs[iter->second] = value;
}

//
std::vector<double> TxAttributeSet::getPrmVec( const std::string& prmVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = prmVecIndices->find(prmVecName);
  if ( iter == prmVecIndices->end() ) throw TxDebugExcept(std::string("No prmVec named '") + prmVecName + "'");
  return prmVecs[iter->second];
}

//
int TxAttributeSet::getPrmVecIndex(const std::string& prmVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = prmVecIndices->find(prmVecName);
  if ( iter != prmVecIndices->end() ) return iter->second;
  else return getNumPrmVecs();
}

// Append an prmVec of this name and the given value
bool TxAttributeSet::appendPrmVec(const std::string& prmVecName, const std::vector<double>& value) 
{
  if (publicReshape) return appendPrmVecNoChk(prmVecName, value);
  else return false;
}

//
std::pair< std::string, std::vector<double> >  TxAttributeSet::getCurrentPrmVecAndBump() const 
{
  std::pair<std::string, std::vector<double> > p(prmVecIter->first, prmVecs[prmVecIter->second]);
  prmVecIter++;
  if (prmVecIter==prmVecIndices->end()) prmVecIter = prmVecIndices->begin();
  return p;
}

//
// StrVec accessors
//

//
bool TxAttributeSet::hasStrVec(const std::string& strVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = strVecIndices->find(strVecName);
  if ( iter == strVecIndices->end() ) 
    {
      return false;  // Disabling exception for now.  Compatibility issue with vorpal.
      std::cerr<< "No string vector  named: "<<strVecName<<std::endl; 
      TxDebugExcept txde("No string vector named: ");
      txde <<strVecName;
      throw txde;
    }
  return true;
}

//
void TxAttributeSet::setStrVec(const std::string& strVecName, const std::vector<std::string>& value)
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = strVecIndices->find(strVecName);
  if ( iter != strVecIndices->end() ) strVecs[iter->second] = value;
}

//
std::vector<std::string> TxAttributeSet::getStrVec( const std::string& strVecName) const 
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = strVecIndices->find(strVecName);
  if ( iter == strVecIndices->end() ) throw TxDebugExcept(std::string("No strVec named '") + strVecName + "'");
  return strVecs[iter->second];
}

//
int TxAttributeSet::getStrVecIndex(const std::string& strVecName) const
{
  std::map<std::string, int, std::less<std::string> >::iterator iter = strVecIndices->find(strVecName);
  if ( iter != strVecIndices->end() ) return iter->second;
  else return getNumStrVecs();
}

// Append an strVec of this name and the given value
bool TxAttributeSet::appendStrVec(const std::string& strVecName, const std::vector<std::string>& value)
{
  if (publicReshape) return appendStrVecNoChk(strVecName, value);
  else return false;
}

//
std::pair< std::string, std::vector<std::string> > TxAttributeSet::getCurrentStrVecAndBump() const
{
  std::pair<std::string, std::vector< std::string> > p(strVecIter->first, strVecs[strVecIter->second]);
  strVecIter++;
  if (strVecIter==strVecIndices->end()) strVecIter = strVecIndices->begin();
  return p;
}


std::vector<int> TxAttributeSet::getIntTypes()
{
  size_t i;
  std::vector<int> itypes;
  for (i=0; i<types.size(); i++)
    {
      int itype = -1;
      std::string tp = types[i];
      if ( tp == "options" )      itype = TX_OPTION_TYPE;
      else if ( tp == "params" )  itype = TX_PARAM_TYPE;
      else if ( tp == "strings" ) itype = TX_STRING_TYPE;
      else if ( tp == "optvecs" ) itype = TX_OPTVEC_TYPE;
      else if ( tp == "prmvecs" ) itype = TX_PRMVEC_TYPE;
      else if ( tp == "strvecs" ) itype = TX_STRVEC_TYPE;
      itypes.push_back(itype);
    }
  return itypes;
}


//
// I/O
//

// int TxAttributeSet::indentation = -1;


void TxAttributeSet::indent(std::ostream& out, size_t indentation) const 
{
  for (size_t i=0; i<indentation; ++i) out << "  ";
}

/*
void TxAttributeSet::write(std::ostream& ostr, bool dump) const 
{
  writeHead(ostr);
  writeBody(ostr, dump);
  writeTail(ostr);
}
//*/

void TxAttributeSet::write(std::ostream& ostr, bool dump, size_t indentation) const 
{
  writeHead(ostr);
  writeBody(ostr, dump, indentation);
  writeTail(ostr);
}


//
void TxAttributeSet::writeHead(std::ostream& ostr, size_t indentation)	const 
{
  indent(ostr, indentation);
  ostr << '<';
  if (type.length()) ostr << type;
  else ostr << getClassName();
  ostr << " " << getObjectName() << '>' << std::endl;
}

//
void TxAttributeSet::writeBody(std::ostream& ostr, bool dump,size_t indentation) const 
{
  //++indentation;

  std::map<std::string, int, std::less<std::string> >::iterator iter;
  
  if (optionIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<options>" << std::endl;
    for (iter=optionIndices->begin(); iter!=optionIndices->end(); iter++){
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      ostr << iter->first << " = " << options[iter->second] << std::endl;
    }
    if (dump) indent(ostr, indentation), ostr << "</options>" << std::endl;
  }
  
  if (paramIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<params>" << std::endl;
    for (iter=paramIndices->begin(); iter!=paramIndices->end(); iter++){
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      ostr << iter->first << " = " << params[iter->second] << std::endl;
    }
    if (dump) indent(ostr, indentation), ostr << "</params>" << std::endl;
  }
  
  if (stringIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<strings>" << std::endl;
    for (iter=stringIndices->begin(); iter!=stringIndices->end(); iter++) {
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      //ostr << iter->first << " = " << '"' << strings[iter->second] << '"' << std::endl;
      ostr << iter->first << " = "  << strings[iter->second] << std::endl;  //modified by wy 2008.12.26
    }
    if (dump) indent(ostr, indentation), ostr << "</strings>" << std::endl;
  }
  
  if (optVecIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<optvecs>" << std::endl;
    for (iter=optVecIndices->begin(); iter!=optVecIndices->end(); iter++)  {
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      ostr << iter->first << " = [";
      if (optVecs[iter->second].size()) ostr << optVecs[iter->second][0];
      for (size_t i=1; i<optVecs[iter->second].size(); ++i) ostr << " " << optVecs[iter->second][i];
      ostr <<" "<< ']' << std::endl;
    }
    if (dump) indent(ostr, indentation), ostr << "</optvecs>" << std::endl;
  }
  
  if (prmVecIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<prmvecs>" << std::endl;
    for (iter=prmVecIndices->begin(); iter!=prmVecIndices->end(); iter++) {
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      ostr << iter->first << " = [";
      if (prmVecs[iter->second].size()) ostr << prmVecs[iter->second][0];
      for (size_t i=1; i<prmVecs[iter->second].size(); ++i) ostr << " " << prmVecs[iter->second][i];
      ostr <<" "<< ']' << std::endl;
    }
    if (dump) indent(ostr, indentation), ostr << "</prmvecs>" << std::endl;
  }
  
  if (strVecIndices->size())  {
    if (dump) indent(ostr, indentation), ostr << "<strvecs>" << std::endl;
    for (iter=strVecIndices->begin(); iter!=strVecIndices->end(); iter++) {
      indent(ostr, indentation);
      if(dump) indent(ostr, indentation);
      ostr << iter->first << " = [";
      if (strVecs[iter->second].size()) ostr << '"'<<strVecs[iter->second][0]<<'"';
      for (size_t i=1; i<strVecs[iter->second].size(); ++i)  ostr << " " <<'"' << strVecs[iter->second][i] << '"';
      //for (size_t i=1; i<strVecs[iter->second].size(); ++i)  ostr << " " << strVecs[iter->second][i] ;
      ostr <<" "<< ']' << std::endl;
    }
    if (dump) indent(ostr, indentation), ostr << "</strvecs>" << std::endl;
  }
}

//
void TxAttributeSet::writeTail(std::ostream& ostr, size_t indentation)const
{
  indent(ostr, indentation);
  ostr << "</";
  if (type.length()) ostr << type;
  else ostr << getClassName();
  ostr << ">" << std::endl;
}

//
std::string TxAttributeSet::read(std::istream& istr)
{
  // Read a line at a time.
  // Ignore anything on a line past '#'
  // Lines starting with <options>, <params>, <strings>, <othertype name>
  //  invoke ReadType(std::string), which quits at </options>, etc.
  // Other lines starting with '</' cause quit
  // Return line that caused end of reading if does not begin with </
  //   otherwise return empty line

  lineNum = 0;
  std::string line;
  // Loop to get all lines
  while (!istr.eof()) {
    line = TxAttributeSet::trimLine(getLine(istr), commentChar);
    if ( ! line.length() ) continue;

    int startIndex, endIndex;
    
    // Determine whether noting a type
    if (line[0]=='<') {
      // Look for end
      endIndex = line.find('>');   if (endIndex == -1) return line;
      // Determine type
      startIndex = line.find_first_not_of(" \t", 1);  if (startIndex == endIndex) return line;
      if (line[startIndex] == '/' ) return "";
      endIndex = line.find_last_not_of(" \t", endIndex-1) + 1;
      
      std::string tp = line.substr(startIndex, endIndex-startIndex);
      std::string retline = readType(istr, tp);
      
      if (retline.length()) return retline;
    }
    else{ // Named variable, so read
      if ( !parseLine(line) ) std::cerr << "Line '" << line <<	"' not parsed.\n";
    }
  }
  return "";
}

//
std::string TxAttributeSet::readType(std::istream& istr, const std::string& tp) 
{
  // Determine type.  If not known, return.
  int itype = -1;
  if ( tp == "options" ) itype = TX_OPTION_TYPE;
  else if ( tp == "params"  ) itype = TX_PARAM_TYPE;
  else if ( tp == "strings" ) itype = TX_STRING_TYPE;
  else if ( tp == "optvecs" ) itype = TX_OPTVEC_TYPE;
  else if ( tp == "prmvecs" ) itype = TX_PRMVEC_TYPE;
  else if ( tp == "strvecs" ) itype = TX_STRVEC_TYPE;
  else return tp;

// Loop to get all lines
  std::string line;
  while (!istr.eof()) {
    line = getLine(istr);
    std::string trimline = TxAttributeSet::trimLine(line, commentChar);

    //if ( trimline[0] == '/') return ""; //original code, the following codes is used to correct this one
    /***************************************************************************************/
    //find '</' to check whether this attribute is finished=============================>>>>
    /***************************************************************************************/
    if ( trimline[0]=='<') {  
      int startIndex,endIndex;
      // Look for end index
      endIndex = trimline.find('>');  if (endIndex == -1) return "";
      // Look for first index after '>'
      startIndex = trimline.find_first_not_of(" \t", 1);  if (startIndex == endIndex) return "";
      // trimLine[startIndex] = '/', mean this attribute with Type of "tp" is finish 
      if (trimline[startIndex] == '/' ) return "";
    }
    /***************************************************************************************/
    //find '</' to check whether this attribute is finished=============================<<<<
    /***************************************************************************************/

    // get name and value
    std::pair<std::string, std::string> nv = getNameAndValueStr(trimline);
    if ( nv.second == "" ) continue;


    ///////////////////// Look for name in appropriate type/////////////////////
    switch (itype) 
      {
      case TX_OPTION_TYPE:
	{
	  bool hasOpt = hasOption(nv.first);
	  if (hasOpt || publicReshape )
	    {
	      int d;
	      // Allow for bools
	      if (nv.second == "true") d = 1;
	      else if (nv.second == "false") d = 0;
	      else d = (int) strtol(nv.second.c_str(), NULL, 10);
	      if (hasOpt) setOption(nv.first, d);
	      else appendOptionNoChk(nv.first, d);
	    }
	}
	break;

      case TX_PARAM_TYPE:
	{
	  bool hasPar = hasParam(nv.first);
	  if (hasPar || publicReshape ) {
	    double d;
	    d = strtod(nv.second.c_str(), NULL);
	    if (hasPar) setParam(nv.first, d);
	    else appendParamNoChk(nv.first, d);
	  }
	}
	break;
	
      case TX_STRING_TYPE:
	{
	  if (hasString(nv.first)) setString(nv.first, nv.second);
	  else if (publicReshape) appendStringNoChk(nv.first, nv.second);
	}
	break;
      case TX_OPTVEC_TYPE:
	{
	  bool hasOV = hasOptVec(nv.first);
	  if ( hasOV || publicReshape )  {
	    ///////////////////////////// Remove brackets////////////////////////////////////////////////////////////////
	    //std::istringstream sstr(nv.second.substr(1, nv.second.length()-1)); //original codes
	    std::istringstream sstr(nv.second.substr(1, nv.second.length()-2));   // modified by wy 2008.12.26
	    
	    std::string s;
	    sstr >> s;
	    int d;
	    char* endptr;
	    std::vector<int> vals;
	    while (s.length()){
	      d = (int) strtol(s.c_str(), &endptr, 10);
	      if (*endptr == '\0') vals.push_back(d);
	      s = "";
	      sstr >> s;
	    }
	    if (hasOV) setOptVec(nv.first, vals);
	    else appendOptVecNoChk(nv.first, vals);
	  }
	}
	break;
	
      case TX_PRMVEC_TYPE:
	{
	  bool hasPV = hasPrmVec(nv.first);
	  if (hasPV || publicReshape ) {
	    ///////////////////////////// Remove brackets////////////////////////////////////////////////////////////////		
	    //std::istringstream sstr(nv.second.substr(1, nv.second.length()-1));  //original codes
	    std::istringstream sstr(nv.second.substr(1, nv.second.length()-2));    // modified by wy 2008.12.26
	    
	    std::string s;
	    sstr >> s;
	    double dd;
	    char* endptr;
	    std::vector<double> vals;
	    while (s.length()) {
	      dd = strtod(s.c_str(), &endptr);
	      if (*endptr == '\0') vals.push_back(dd);
	      s = "";
	      sstr >> s;
	    }
	    if (hasPV) setPrmVec(nv.first, vals);
	    else appendPrmVecNoChk(nv.first, vals);
	  }
	}
	break;
	
      case TX_STRVEC_TYPE:
	{
	  bool hasSV = hasStrVec(nv.first);
	  std::vector<std::string> vals;
	  if (hasSV || publicReshape ) {
	    //////////////////////////////Remove brackets////////////////////////////////////////////////////////////////
	    //std::istringstream sstr(nv.second.substr(1, nv.second.length()-1));  //original codes
	    std::istringstream sstr(nv.second.substr(1, nv.second.length()-2));    // modified by wy 2008.12.26

	    readQuotedStringVecs(sstr, vals);
	    
	    if (hasSV) setStrVec(nv.first, vals);
	    else appendStrVecNoChk(nv.first, vals);
	  }
	}
	break; 
      }
  }
  return "";
}

void TxAttributeSet::readQuotedStringVecs(istream& sstr, vector<string>& vals)
{
  std::string s;
  s="";

  char c;
  bool start_quote;

  start_quote = false;

  while ((c = sstr.peek()))
    {
      if (c == ']' || sstr.eof())
	{
	  sstr.ignore();
	  if (!s.empty()) 
	    {
	      vals.push_back(s);
	    }
	  break;
	}
      
      else if ((c == ' ') || (c == '\t'))
	{
	  if (start_quote)
	    {
	      s += sstr.get();
	    }
	  else {
	    if (!s.empty()) 
	      {
		vals.push_back(s);
	      }
	    s = "";
	    sstr.ignore();
	  }
	}
      
      else if (c == '"')
	{
	  sstr.ignore(); // we don't want to save the quote itself.
	  
	  if (start_quote == true)
	    { // closing the quote.
	      start_quote = false;
	      if (!s.empty())
		{
		  vals.push_back(s);
		}
	      s= "";
	    }
	  
	  else if (start_quote == false)
	    {
	      start_quote = true;
	      if (!s.empty()) 
		{
		  vals.push_back(s);
		}
	      s = "";
	    }
	}
      
      else 
	{  // regular char... read it.
	  s += sstr.get();
	}
    }
}

bool TxAttributeSet::parseLine(const std::string& line)
{
  ///////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////// get name and value and try to figure out what this is/////////////////////

  std::string trimline = TxAttributeSet::trimLine(line, commentChar);

  if ( ! trimline.length() ) return true;
  std::pair<std::string, std::string> nv = getNameAndValueStr(trimline);
  
  if (nv.second[0] == '"' ) {
    nv.second = nv.second.substr(1, nv.second.length()-2);
    goto isstring;
  }

  ///////////////////////////////////////////////////////////////////////
  /////////////////////////////// Try as a bool//////////////////////////

  if ( nv.second == "true" ) {
    if (hasOption(nv.first)) setOption(nv.first, 1);
    else if (publicReshape) appendOptionNoChk(nv.first, 1);
    return true;
  }
  if ( nv.second == "false" )  {
    if (hasOption(nv.first)) setOption(nv.first, 0);
    else if (publicReshape) appendOptionNoChk(nv.first, 0);
    return true;
  }
  
  ///////////////////////////////////////////////////////////////////
  ///////////////////// Try as a vector//////////////////////////////
  if (nv.second[0] == '[') {
    ////////////////////////// Get the vector of values and put it into a stream////////////
    std::string vecvals = nv.second.substr(1, nv.second.length()-2);
    vecvals = TxAttributeSet::trimLine(vecvals, commentChar);
    
    ////////////////////////// Append empty vector as a vector of strings///////////////////
    if (vecvals.size() == 0) {
      if (hasStrVec(nv.first) || publicReshape) {
	std::vector<std::string> no_vals(0);
	if (hasStrVec(nv.first)) setStrVec(nv.first, no_vals);
	else appendStrVecNoChk(nv.first, no_vals);
      }
      return true;
    }

    std::istringstream sstr(vecvals);
    
    ///////////////////////// Pop the strings one by one//////////////////////////////////
    std::string s;
    sstr >> s;
    
    // If first is an integer, then create a vector of integers
    char* endptr;
    int d = (int) strtol(s.c_str(), &endptr, 10);
    if (*endptr == '\0') {
      if (hasOptVec(nv.first) || publicReshape) {
	std::vector<int> vals;
	vals.push_back(d);
	s = "";
	sstr >> s;
	while (s.length()) {
	  d = (int) strtol(s.c_str(), &endptr, 10);
	  if (*endptr == '\0') vals.push_back(d);
	  s = "";
	  sstr >> s;
	}
	if (hasOptVec(nv.first)) setOptVec(nv.first, vals);
	else appendOptVecNoChk(nv.first, vals);
      }
      return true;
    }
    
    // If first is a double, create a vector of doubles
    double dd = (double) strtod(s.c_str(), &endptr);
    if (*endptr == '\0') {
      if (hasPrmVec(nv.first) || publicReshape){
	std::vector<double> vals;
	vals.push_back(dd);
	s = "";
	sstr >> s;
	while (s.length()){
	  dd = (double) strtod(s.c_str(), &endptr);
	  if (*endptr == '\0') vals.push_back(dd);
	  s = "";
	  sstr >> s;
	}
	if (hasPrmVec(nv.first)) setPrmVec(nv.first, vals);
	else appendPrmVecNoChk(nv.first, vals);
      }
      return true;
    }
    
    // If first is a string, read as a vector of strings
    if (nv.second[nv.second.length()-1] == ']'){
      if (hasStrVec(nv.first) || publicReshape) {
	std::istringstream mysstr(vecvals); // making a new one here since the first item already read in sstr.
	std::vector<std::string> vals;
	readQuotedStringVecs(mysstr, vals);
	
	if (hasStrVec(nv.first)) setStrVec(nv.first, vals);
	else appendStrVecNoChk(nv.first, vals);
      }
      return true;
    }
    
    // Not a vector; must be a string
    goto isstring;
  }
  
  // Try as an int
  {
    char* endptr;
    int d = (int) strtol(nv.second.c_str(), &endptr, 10);
    if (*endptr == '\0') {
      // Worked
      if (hasOption(nv.first)) setOption(nv.first, d);
      else if (hasParam(nv.first)) setParam(nv.first, d);
      else if (publicReshape) appendOptionNoChk(nv.first, d);
      return true;
    }
  }
  
  // Try as a double
  {
    char* endptr;
    double dd = (double) strtod(nv.second.c_str(), &endptr);
    if (*endptr == '\0'){
      // Worked
      if (hasParam(nv.first)) setParam(nv.first, dd);
      else if (publicReshape) appendParamNoChk(nv.first, dd);
      return true;
    }
  }
  
  // Then must be a string
 isstring:
  bool hasStr = hasString(nv.first);
  if (hasStr) setString(nv.first, nv.second);
  else {
    if (publicReshape) appendStringNoChk(nv.first, nv.second);
  }
  return true;
}

/******************************************************************************/
/*
 * static Function
 * Find the indexes of first and last nonwhite character
 * and get the useful substring by using the indexes
*/
/******************************************************************************/
std::string TxAttributeSet::trimLine(const std::string& line, char commentchar) 
{
  // Find index of first nonwhite character.  If none or backslash, return.
  bool hasComment = false;
  int startIndex, endIndex, tmpIndex;
  startIndex = line.find_first_not_of(" \t", 0);
  if ( startIndex == -1 )  return "";
  
  // Find comment character or end of line.
  endIndex = line.find(commentchar, 0);
  
  if (endIndex == -1) endIndex = line.length();
  else {
    hasComment = true;
    if (endIndex ==0) return "";
    else endIndex --;
  }

  // Find last nonwhite character, create substring
  tmpIndex = endIndex;
  if (hasComment==false) {
    endIndex = line.find_last_not_of(" \t", tmpIndex - 1);
  }
  else {
    endIndex = line.find_last_not_of(" \t", endIndex); 
  }
  if (endIndex == 0) endIndex = tmpIndex;
  else endIndex++;
  
  // empty line if start of line is larger than end of line (comment)
  if (startIndex > endIndex) {
    return "";
  }

  std::string trimline = line.substr(startIndex, endIndex - startIndex);
  return trimline;
}


/***************************************************************************************/
/***************************************************************************************/
std::string TxAttributeSet::getAnyPlatformLine(std::istream& istr,char commentchar) 
{
  std::string line("");
  // The following section replaces the system getline method.
  char c='\0';
  while ( 1 ) {
    if ( !istr.eof() ){  // optimization: tests true most of the time.
      istr.get(c);
    }
    else{
      return "";        // eof return empty string.
    }
    if ((c != '\r') && (c != '\n')) {
      line += c;
    }
    else if (c == '\r') {
      if (istr.peek() == '\n')  istr.ignore();
      break;
    }
    else if (c == '\n') {  // \n
      break;
    }
  }
  line = TxAttributeSet::trimLine(line, commentchar);
  return line;
}

/******************************************************************************/
/*
 * member Function: getLine
 * use to getLine from istream
 * when meet '\\', connect the current line with the next line
 * when meet the number of ']' is less than the number of '[', connect the current line with the next line
 *
 *
 * when meet the number of ']' is greater than the number '[', throw an error
 * when the number of '[' is greater than 1, throw an error
*/
/******************************************************************************/
std::string TxAttributeSet::getLine(std::istream& istr) 
{
  std::string line;
// Get the first line
  if (istr.eof()) return "";
  
  line = getAnyPlatformLine(istr, commentChar);

  lineNum++;

  if (!line.length()) return line;


  // Allow for continuation with '\'
  bool backSlashEnd = (line[line.size() - 1] == '\\');
  if (backSlashEnd) line = line.substr(0, line.size() - 1);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////// Check for bracket sanity////////////////////////////////////////////////////////

  size_t numLeftBrackets = getNumChars(line, '[');
  size_t numRightBrackets = getNumChars(line, ']');

  // cerr << "TxAttributeSet::getLine: brackets counted: " << numLeftBrackets << " left and " << numRightBrackets << " right.\n";

  if (numLeftBrackets > 1)  {
    TxDebugExcept txde("More than one left bracket at line ");
    txde << lineNum;
    throw txde;
  }
  if (numRightBrackets > numLeftBrackets)  {
    TxDebugExcept txde("More right brackets than left at line ");
    txde << lineNum;
    throw txde;
  }
  // cerr << "TxAttributeSet::getLine: passed bracket test.\n";
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////Continue reading until brackets close and not ended with a backslash////////////////

  while (numRightBrackets < numLeftBrackets || backSlashEnd) {
    // Get next line
    std::string nextline = getAnyPlatformLine(istr, commentChar);
    lineNum++;
    backSlashEnd = (nextline[nextline.size() - 1] == '\\');
    if (backSlashEnd){
      nextline = nextline.substr(0, nextline.size() - 1);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// Check for bracket sanity/////////////////////////////////////////////////////
    
    numLeftBrackets += getNumChars(nextline, '[');
    numRightBrackets += getNumChars(nextline, ']');
    if (numLeftBrackets > 1) {
      TxDebugExcept txde("More than one left bracket at line ");
      txde << lineNum;
      throw txde;
    }
    if (numRightBrackets > numLeftBrackets){
      TxDebugExcept txde("More right brackets than left at line ");
      txde << lineNum;
      throw txde;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////Add to line//////////////////////////////////////////////////////////////
    
    line += nextline; 
  }
  return line;
}

//
std::pair<std::string, std::string>TxAttributeSet::getNameAndValueStr(std::string line)
{
  // Find name string
  int equalsIndex = line.find('=', 0);
  if ( equalsIndex == -1 ) 
    return std::pair<std::string, std::string>("", "");

  int endNameIndex = line.find_last_not_of(" \t", equalsIndex - 1) + 1;
  std::string nameStr = line.substr(0, endNameIndex);
  
  // Find value string
  int startIndex = line.find_first_not_of(" \t", equalsIndex + 1);
  if ( startIndex == -1 ) 
    return std::pair<std::string, std::string>("", "");

  int endIndex = line.length();
  std::string valueStr = line.substr(startIndex, endIndex);
  //int indexNoQ = valueStr.length()-1;

  return std::pair<std::string, std::string>(nameStr, valueStr);
}


//
// Appending new data
//

//
bool TxAttributeSet::appendOptionNoChk(const std::string& optionName, int value)
 {
  options.push_back(value);
  names.push_back(optionName);
  types.push_back("options");
  bool res = optionIndices->insert(std::pair<const std::string, int>(optionName, options.size()-1)).second;
  if (!res)
    {	// Cannot add second key of same type
      options.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstOption();
  return true;
}

//
bool TxAttributeSet::appendParamNoChk(const std::string& paramName, double value) 
{
  params.push_back(value);
  names.push_back(paramName);
  types.push_back("params");
  bool res = paramIndices->insert(std::pair<const std::string, int>(paramName, params.size()-1)).second;
  if (!res) 
    {	// Cannot add second key of same type
      params.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstParam();
  return true;
}

//
bool TxAttributeSet::appendStringNoChk(const std::string& stringName, const std::string& value)
{
  strings.push_back(value);
  names.push_back(stringName);
  types.push_back("strings");
  bool res = stringIndices->insert(std::pair<const std::string, int>(stringName, strings.size()-1)).second;
  if (!res)
    {	// Cannot add second key of same type
      strings.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstString();
  return true;
}

//
bool TxAttributeSet::appendOptVecNoChk(const std::string& optVecName, const std::vector<int>& value) 
{
  optVecs.push_back(value);
  names.push_back(optVecName);
  types.push_back("optvecs");
  bool res = optVecIndices->insert( std::pair<const std::string, int >( optVecName, optVecs.size()-1)).second;
  if (!res) 
    {	// Cannot add second key of same type
      optVecs.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstOptVec();
  return true;
}

//
bool TxAttributeSet::appendPrmVecNoChk(const std::string& prmVecName, const std::vector<double>& value) 
{
  prmVecs.push_back(value);
  names.push_back(prmVecName);
  types.push_back("prmvecs");
  bool res = prmVecIndices->insert( std::pair<const std::string, int >( prmVecName, prmVecs.size()-1)).second;
  if (!res) 
    {	// Cannot add second key of same type
      prmVecs.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstPrmVec();
  return true;
}

//
bool TxAttributeSet::appendStrVecNoChk(const std::string& strVecName, const std::vector<std::string>& value) 
{
  strVecs.push_back(value);
  names.push_back(strVecName);
  types.push_back("strvecs");
  bool res = strVecIndices->insert( std::pair<const std::string, int >(strVecName, strVecs.size()-1)).second;
  if (!res) 
    {	// Cannot add second key of same type
      strVecs.pop_back();
      names.pop_back();
      types.pop_back();
      return false;
    }
  setToFirstStrVec();
  return true;
}

//
void TxAttributeSet::emptyNoChk() 
{
  if (!publicReshape) return;
  if (optionIndicesOwner) 
    {
      delete optionIndices;
      optionIndices = new std::map< std::string, int, std::less<std::string> >;
      options.erase(options.begin(), options.end());
    }
  if (paramIndicesOwner) 
    {
      delete paramIndices;
      paramIndices = new std::map< std::string, int,  std::less<std::string> >;
      params.erase(params.begin(), params.end());
    }
  if (stringIndicesOwner) 
    {
      delete stringIndices;
      stringIndices = new std::map< std::string, int, std::less<std::string> >;
      strings.erase(strings.begin(), strings.end());
    }
  if (optVecIndicesOwner)
    {
      delete optVecIndices;
      optVecIndices = new std::map< std::string, int,  std::less<std::string> >;
      optVecs.erase(optVecs.begin(), optVecs.end());
    }
  if (prmVecIndicesOwner)
    {
      delete prmVecIndices;
      prmVecIndices = new std::map< std::string, int,  std::less<std::string> >;
      prmVecs.erase(prmVecs.begin(), prmVecs.end());
    }
  if (strVecIndicesOwner)
    {
      delete strVecIndices;
      strVecIndices = new std::map< std::string, int,  std::less<std::string> >;
      strVecs.erase(strVecs.begin(), strVecs.end());
    }
  names.clear();
  types.clear();
}

//
// Initializing
//
void TxAttributeSet::setup()
{
  optionIndices = new std::map< std::string, int, std::less<std::string> >;
  optionIndicesOwner = true;
  optionIter = optionIndices->begin();
  paramIndices = new std::map< std::string, int, std::less<std::string> >;
  paramIndicesOwner = true;
  paramIter = paramIndices->begin();
  stringIndices = new std::map< std::string, int, std::less<std::string> >;
  stringIndicesOwner = true;
  stringIter = stringIndices->begin();
  optVecIndices = new std::map< std::string, int, std::less<std::string> >;
  optVecIndicesOwner = true;
  optVecIter = optVecIndices->begin();
  prmVecIndices = new std::map< std::string, int, std::less<std::string> >;
  prmVecIndicesOwner = true;
  prmVecIter = prmVecIndices->begin();
  strVecIndices = new std::map< std::string, int, std::less<std::string> >;
  strVecIndicesOwner = true;
  strVecIter = strVecIndices->begin();
  commentChar = '#';
// I/O
}

std::string TxAttributeSet::getClassName() const 
{
  return "TxAttributeSet";
}
std::string TxAttributeSet::getBaseClass() const
{
  return "TxAttributeSet";
}
//
// Static data
//
//const std::string TxAttributeSet::className("TxAttributeSet");
//const std::string TxAttributeSet::baseClassName("TxAttributeSet");

