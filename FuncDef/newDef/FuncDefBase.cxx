#include <FuncDefBase.hxx>


TxDblFormulaList* FuncDefBase::m_global_dfl = NULL;
std::map< std::string, double, std::less<std::string> > FuncDefBase::m_global_params;


FuncDefBase::
FuncDefBase()
{

}

FuncDefBase::
~FuncDefBase()
{

}

void 
FuncDefBase::
setAttrib_Param(const TxHierAttribSet& tas)
{
  m_params.clear();

  std::vector<std::string>::iterator iter;
  std::vector<std::string> thePrmNames = tas.getStrVec("params");
  for(iter = thePrmNames.begin(); iter!=thePrmNames.end(); iter++){
    string currName = *iter;
    double currValue = 0.;
    if(tas.hasParam(currName) ){
      currValue = tas.getParam(currName);
     }
    m_params.insert( std::pair<const std::string, double>(currName, currValue) );
  }

  // need to cheeck the global param's name wheather which is same with the name of one m_params! 
  std::map< std::string, double, std::less<std::string> >theGlobalVariables;
  GetGlobalVariables(theGlobalVariables);

  std::map< std::string, double, std::less<std::string> >::const_iterator dIter;
  for(dIter = theGlobalVariables.begin(); dIter!=theGlobalVariables.end(); dIter++){
    m_params.insert(*dIter);
  }
}


void 
FuncDefBase::
setAttrib_DFL(const TxHierAttribSet& tas)
{
  TxDblFormulaList* m_dfl = new TxDblFormulaList();

  std::vector<std::string> m_Exps = tas.getStrVec("formulas");
  std::vector<std::string>::iterator iter; 

  for(iter = m_Exps.begin(); iter!=m_Exps.end(); iter++){
    string currName = *iter;
    if(tas.hasString(currName)){
      string currExpression = tas.getString(currName);
      try {
	m_dfl->append(TxDblFormula(currName, currExpression));
      } catch (TxDebugExcept& tde){
	std::cerr << "FuncDefBase::setAttrib exception: " << tde << std::endl;
	//exit(1);
      }
    }else{
      std::cerr << "FuncDefBase::setAttrib no expression :"<< currName << "defined" << std::endl;
    }
  }
}



void 
FuncDefBase::
combine_DFL_Params()
{
  m_dfl->setParams(m_params);
}



void 
FuncDefBase::
setGlobalAttrib(const TxHierAttribSet& tha)
{
  std::vector< std::string > globalVariableDefines = tha.getNamesOfType("GlobalVariables");
  if(globalVariableDefines.empty()){
    return;
  }else if(globalVariableDefines.size()>1){
    std::cout << "FuncDefBase::setGlobalAttrib-------- warning---only one group GlobalVariables is valid"<<endl;;
  }

  if(m_global_dfl!=NULL) delete m_global_dfl;
  m_global_dfl = new TxDblFormulaList();

  TxHierAttribSet tas = tha.getAttrib(globalVariableDefines[0]);

  m_global_params.clear();
  std::vector<std::string>::iterator iter;
  std::vector<std::string> thePrmNames = tas.getStrVec("global_params");
  for(iter = thePrmNames.begin(); iter!=thePrmNames.end(); iter++){
    string currName = *iter;
    double currValue = 0.;
    if(tas.hasParam(currName) ){
      currValue = tas.getParam(currName);
     }
    m_global_params.insert( std::pair<const std::string, double>(currName, currValue) );
  }

  TxDblFormulaList* m_dfl = new TxDblFormulaList();

  std::vector<std::string> m_Exps = tas.getStrVec("global_formulas");
  for(iter = m_Exps.begin(); iter!=m_Exps.end(); iter++){
    string currName = *iter;
    if(tas.hasString(currName)){
      string currExpression = tas.getString(currName);
      m_global_dfl->append(TxDblFormula(currName, currExpression));
      try {
	m_global_dfl->append(TxDblFormula(currName, currExpression));
      } catch (TxDebugExcept& tde){
	std::cerr << "FuncDefBase::setGlobalAttrib exception: " << tde << std::endl;
	//exit(1);
      }
    }else{
      std::cerr << "FuncDefBase::setAttrib no expression :"<< currName << "defined" << std::endl;
    }
  }

  m_global_dfl->setParams(m_global_params);
  m_global_dfl->evaluate();
}



void 
FuncDefBase::
GetGlobalVariables(std::map< std::string, double, std::less<std::string> >& theGlobalVariables)
{
  theGlobalVariables.clear();
  if(m_global_dfl!=NULL){
    std::map< std::string, double, std::less<std::string> > static_result;
    m_global_dfl->getResult(theGlobalVariables);
  }
}
