
#include <ExpressionFuncDef.hxx>

TxDblFormulaList* ExpressionFuncDef::m_global_dfl = NULL;
std::map< std::string, double, std::less<std::string> > ExpressionFuncDef::m_global_params;

ExpressionFuncDef::
ExpressionFuncDef()
{
  TxDblFormulaList* m_dfl = NULL;
}

ExpressionFuncDef::
~ExpressionFuncDef()
{
  m_params.clear();
  if(m_dfl!=NULL) delete m_dfl;
}

void 
ExpressionFuncDef::
setAttrib(const TxHierAttribSet& tas)
{
  std::vector<std::string>::iterator iter;
  std::vector<std::string> m_Prms = tas.getStrVec("params");
  for(iter = m_Prms.begin(); iter!=m_Prms.end(); iter++){
    string currName = *iter;
    double currValue = 0.;
    if(tas.hasParam(currName) ){
      currValue = tas.getParam(currName);
     }
    m_params.insert( std::pair<const std::string, double>(currName, currValue) );
  }

  // need to cheeck the global param's name wheather which is same with the name of one m_params! 
  if(m_global_dfl!=NULL){
    std::map< std::string, double, std::less<std::string> > static_result;
    m_global_dfl->getResult(static_result);

    std::map< std::string, double, std::less<std::string> >::const_iterator dIter;
    for(dIter = static_result.begin(); dIter!=static_result.end(); dIter++){
      m_params.insert(*dIter);
    }
  }


  TxDblFormulaList* m_dfl = new TxDblFormulaList();

  std::vector<std::string> m_Exps = tas.getStrVec("formulas");
  for(iter = m_Exps.begin(); iter!=m_Exps.end(); iter++){
    string currName = *iter;
    if(tas.hasString(currName)){
      string currExpression = tas.getString(currName);
      try {
	m_dfl->append(TxDblFormula(currName, currExpression));
      } catch (TxDebugExcept& tde){
	std::cerr << "ExpressionFuncDef::setAttrib exception: " << tde << std::endl;
	//exit(1);
      }
    }else{
      std::cerr << "ExpressionFuncDef::setAttrib no expression :"<< currName << "defined" << std::endl;
    }
  }

  m_dfl->setParams(m_params);
}



void 
ExpressionFuncDef::
setGlobalAttrib(const TxHierAttribSet& tha)
{
  std::vector< std::string > globalVariableDefines = tha.getNamesOfType("GlobalVariables");
  if(globalVariableDefines.empty()){
    return;
  }else if(globalVariableDefines.size()>1){
    std::cout << "ExpressionFuncDef::setGlobalAttrib-------- warning---only one group GlobalVariables is valid"<<endl;;
  }

  if(m_global_dfl!=NULL) delete m_global_dfl;
  m_global_dfl = new TxDblFormulaList();


  TxHierAttribSet tas = tha.getAttrib(globalVariableDefines[0]);

  m_global_params.clear();
  std::vector<std::string>::iterator iter;
  std::vector<std::string> m_Prms = tas.getStrVec("global_params");
  for(iter = m_Prms.begin(); iter!=m_Prms.end(); iter++){
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
	std::cerr << "ExpressionFuncDef::setGlobalAttrib exception: " << tde << std::endl;
	//exit(1);
      }
    }else{
      std::cerr << "ExpressionFuncDef::setAttrib no expression :"<< currName << "defined" << std::endl;
    }
  }

  m_global_dfl->setParams(m_global_params);
  m_global_dfl->evaluate();
}



void 
ExpressionFuncDef::
GetGlobalVariables(std::map< std::string, double, std::less<std::string> >& theGlobalVariables)
{
  theGlobalVariables.clear();
  if(m_global_dfl!=NULL){
    std::map< std::string, double, std::less<std::string> > static_result;
    m_global_dfl->getResult(theGlobalVariables);
  }
}
