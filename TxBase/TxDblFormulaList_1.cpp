#include <TxDblFormulaList.h>
#include <TxDblFormula.h>

void 
TxDblFormulaList::
getResult(std::map< std::string, double, std::less<std::string> >& resultData)
{
  resultData.clear();
  std::map< std::string, double, std::less<std::string> >::const_iterator dIter;
  for(dIter = params->begin(); dIter!=params->end(); dIter++){
    resultData.insert(*dIter);
  }

  // need to cheeck the formula's name wheather which is same with the name of one params! 
  size_t numFormula = this->getNumDefinedFormulas();
  std::vector<TxDblFormula>::iterator fIter;
  for(fIter=formulas.begin(); fIter!=formulas.end(); fIter++){
    string currName = fIter->getName();
    size_t currIndex = this->getIndex(currName);
    if(currIndex<numFormula){
      double currResult = formulas[currIndex].getValue(); 
      resultData.insert( std::pair<const std::string, double>(currName, currResult) );
    }
  }
}
