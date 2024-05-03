//-------------------------------------------------------------------
//
// File:        TxDblFormulaList.cpp
//
// Purpose:     Implementation of a list of formulas that one can
//		find by name
//
// Version:     $Id: TxDblFormulaList.cpp 62 2006-09-18 20:19:27Z yew $
//
// Copyright (c) 1996-2003, Tech-X Corporation.  All Rights Reserved.
//
//-------------------------------------------------------------------

// tx includes
#include <TxDblFormulaList.h>

#ifdef DEBUG
#define EVAL_DEBUG
#endif

void TxDblFormulaList::insert(const TxDblFormula& tdf, size_t n) 
{
  tdf.setOutThroughStream(*txout);
  tdf.setErrThroughStream(*txerr);
  if ( !isUnique(tdf.getName()) ) {
    throw TxDebugExcept("TxDblFormulaList::insert: name already known!");
  }
  std::map<std::string, size_t, std::less<std::string> >::iterator mapiter;
  for (mapiter=formulaMap.begin(); mapiter!=formulaMap.end(); ++ mapiter) {
    if (mapiter->second >= n) mapiter->second++;
  }
  std::vector<TxDblFormula>::iterator veciter = formulas.begin();
  veciter += n;
  formulas.insert(veciter, tdf);

  formulaMap.insert( std::pair<const std::string, size_t>(tdf.getName(), n) );
  definedFormulas = (definedFormulas < n ? definedFormulas : n); 
}

void TxDblFormulaList::insertDefined(const TxDblFormula& tdf, size_t n) 
{
  tdf.setOutThroughStream(*txout);
  tdf.setErrThroughStream(*txerr);
  if ( !isUnique(tdf.getName()) ) {
    throw TxDebugExcept("TxDblFormulaList::insertDefined: name already known!");
  }
  std::map<std::string, size_t, std::less<std::string> >::iterator mapiter;
  for (mapiter=formulaMap.begin(); mapiter!=formulaMap.end(); ++ mapiter) {
    if (mapiter->second >= n) mapiter->second++;
  }
  std::vector<TxDblFormula>::iterator veciter = formulas.begin();
  veciter += n;
  formulas.insert(veciter, tdf);
  formulaMap.insert( std::pair<const std::string, size_t>(tdf.getName(), n) );
  // This formula is not defined.  So number defined increase if put into defined region
  definedFormulas = (definedFormulas >= n ? definedFormulas+1 : definedFormulas);
}

void TxDblFormulaList::append(const TxDblFormula& tdf) 
{
  tdf.setOutThroughStream(*txout);
  tdf.setErrThroughStream(*txerr);
  if ( !isUnique(tdf.getName()) ) {
    throw TxDebugExcept("TxDblFormulaList::append: name already known!");
  }
  formulaMap.insert( std::pair<const std::string, size_t>(tdf.getName(), formulas.size()) );
  formulas.push_back(tdf);
}

void TxDblFormulaList::appendDefined(const TxDblFormula& tdf) 
{
  tdf.setOutThroughStream(*txout);
  tdf.setErrThroughStream(*txerr);
  if ( !isUnique(tdf.getName()) ) {
    throw TxDebugExcept("TxDblFormulaList::insert: name already known!");
  }
  std::map<std::string, size_t, std::less<std::string> >::iterator mapiter;
  formulaMap.insert( std::pair<const std::string, size_t>(tdf.getName(), formulas.size()) );
  formulas.push_back(tdf);
  definedFormulas = (definedFormulas == formulas.size()-1 ? formulas.size() : definedFormulas);
}

void TxDblFormulaList::remove(const std::string& nm)
{
  size_t n = formulaMap[nm];
  formulaMap.erase(nm);
  std::map<std::string, size_t, std::less<std::string> >::iterator mapiter;
  for (mapiter=formulaMap.begin(); mapiter!=formulaMap.end(); ++ mapiter) {
    if (mapiter->second > n) mapiter->second--;
  }
  std::vector<TxDblFormula>::iterator veciter = formulas.begin();
  veciter += n;
  formulas.erase(veciter);
  definedFormulas = (definedFormulas < n ? definedFormulas : n);
}

size_t TxDblFormulaList::getIndex(const std::string& nm) const 
{
  std::map<std::string, size_t, std::less<std::string> >::const_iterator mapiter = formulaMap.find(nm);
  if (mapiter == formulaMap.end()) {
    return getNumFormulas();
  }
  return mapiter->second;
}

TxDblFormula TxDblFormulaList::operator[](const std::string& nm) const
{
  std::map<std::string, size_t, std::less<std::string> >::const_iterator mapiter = formulaMap.find(nm);
  if (mapiter == formulaMap.end()) return TxDblFormula();
  return formulas[mapiter->second];
}

TxDblFormula TxDblFormulaList::operator[](size_t i) const 
{
  if (i >= formulas.size()) return TxDblFormula();
  return formulas[i];
}

std::string TxDblFormulaList::getRedefinedNewName(const std::string& nm) const 
{
  std::map<std::string, std::string, std::less<std::string> >::const_iterator mapiter = redefinedNames.find(nm);
  if ( mapiter != redefinedNames.end()) return mapiter->second;
  return std::string();
}

void TxDblFormulaList::replaceRedefinedName(const std::string& nm) 
{
  std::map<std::string, std::string, std::less<std::string> >::iterator mapiter = redefinedNames.find(nm);
  if (mapiter == redefinedNames.end()) return;
  for (size_t j=definedFormulas; j<formulas.size(); ++j) {
    formulas[j].replaceName( mapiter->first, mapiter->second);
  }
}

void TxDblFormulaList::replaceRedefinedNames() 
{
  std::map<std::string, std::string, std::less<std::string> >::iterator mapiter;
  for (mapiter=redefinedNames.begin(); mapiter!=redefinedNames.end(); ++mapiter) 
    replaceRedefinedName(mapiter->first);
}

void TxDblFormulaList::insertRedefinition(const std::string& oldname, const std::string& newname)  
{
  if ( !isUnique(oldname) ) {
    throw TxDebugExcept("TxDblFormulaList::insertRedefinition: name already known!");
  }
  redefinedNames.insert( std::pair<const std::string, std::string>(oldname, newname) );
}

// Find name in neither list
std::string TxDblFormulaList::getUniqueName(const std::string& nm) const 
{
  std::string uname = nm;
  int i;
  for (i=0; i<10000; i++) {
    if (isUnique(nm)) return uname;
    
    std::ostringstream os;
    os << nm << '_' << i;
    uname = os.str();
  }
  return uname;
}

//
// evaluate formulas as possible
//
int TxDblFormulaList::evaluate()
{
  size_t jf, newlyDefForms=1; 

  definedFormulas = 0;

  //  Loop until cannot define
  while (newlyDefForms) {
    newlyDefForms = 0;  //  No forms definable so far
    for (jf=definedFormulas; jf<formulas.size(); jf++) {
      formulas[jf].setMaxEvals(100);
      int evals = formulas[jf].evaluate(*params, *this, definedFormulas);

      if ( evals >= 1) {
#ifdef EVAL_DEBUG
	*txerr <<"definedFormulas = "<< definedFormulas <<"\t jf = "<<jf << "\t Evaluated: "<< formulas[jf] << std::endl;
#endif
        if ( jf != definedFormulas ) {
	  //  Remove formula if name exists in parameters
	  TxDblFormula tdf(formulas[jf]);
          remove(tdf.getName());
          insertDefined(tdf, definedFormulas);
#ifdef EVAL_DEBUG
	  *txerr << "Inserted: " << tdf << std::endl;
#endif
        }else{
	  definedFormulas++;
	}

        newlyDefForms++;
      }

#ifdef EVAL_DEBUG
      else if ( evals < 0 ) {
        *txerr << "TxDblFormulaList::evaluate: unable to define formula " << formulas[jf] << ": '"<< formulas[jf].getLastName() << "\' unknown.\n";
      }
#endif
    }
  }

  // Return number of undefined formulas
  return formulas.size() - definedFormulas;
}

// Print results
void TxDblFormulaList::printEvalResults() const 
{
  //  Print the formulas
  *txout << "\n";
  *txout << formulas.size() << " formulas.\n";
  *txout << definedFormulas << " formulas defined.\n";
  if (definedFormulas) {
    *txout << "Defined formulas:\n";
    for (size_t i=0; i!=definedFormulas; ++i) 
      *txout << formulas[i] << std::endl;
  }
  
  // Print undefined formulas
  *txout << std::endl;
  printUndefined();
  
  //  Print the unknown names
  *txout << std::endl;
  printRedefined();
}

void TxDblFormulaList::printUndefined() const 
{
  if (definedFormulas==formulas.size()) return;

  *txout << "Undefined formulas:\n";
  for (size_t i=definedFormulas; i!=formulas.size(); ++i) 
    *txout << formulas[i] << std::endl;
}

void TxDblFormulaList::printRedefined() const 
{
  if (!redefinedNames.size()) return;

  *txout << "Redefined names:" << std::endl;
  std::map<std::string, std::string, std::less<std::string> >::const_iterator strmapiter;
  for (strmapiter=redefinedNames.begin(); strmapiter!=redefinedNames.end(); ++strmapiter) {
    *txout << strmapiter->first << " = " << strmapiter->second << std::endl;
  }
}

bool TxDblFormulaList::isUnique(const std::string& nm) const 
{
  std::map<std::string, size_t, std::less<std::string> >::const_iterator mapiter = formulaMap.find(nm);
  if (mapiter != formulaMap.end()) return false;
  std::map<std::string, std::string, std::less<std::string> >::const_iterator strmapiter = redefinedNames.find(nm);
  if (strmapiter != redefinedNames.end()) return false;
  if (!params) {
    std::map<std::string, double, std::less<std::string> >::const_iterator prmiter = params->find(nm);
    if (prmiter != params->end()) return false;
  }
  return true;
}

