//--------------------------------------------------------------------
//
// File:	TFuncBase.cxx
//
// Purpose:	Implementation of a functor for a rising with the half-cosine in time.
//
//--------------------------------------------------------------------

#include <TFuncBase.hxx>
#include <PhysConsts.hxx>
#include <TxStreams.h>


TFuncBase::
TFuncBase() : FuncDefBase()
{

}


void 
TFuncBase::
setAttrib_Param(const TxHierAttribSet& tas)
{
  FuncDefBase::setAttrib_Param(tas);
  std::vector<std::string> t_coords = tas.getStrVec("t_pnts");
  std::vector<std::string>::iterator iter;
  for(iter = t_coords.begin(); iter!=t_coords.end(); iter++){
    string currName = *iter;
    if(m_params.find(currName)!=m_params.end()){
      m_TPntVec.push_back(m_params[currName]); 
    }else{
      cout<<"TFuncBase::setAttrib_Param-----------error-------1"<<endl;
    }
  }
}


void 
TFuncBase::
setAttrib_DFL(const TxHierAttribSet& tas)
{
  FuncDefBase::setAttrib_DFL(tas);

  std::vector<std::string> theExpsNames = tas.getStrVec("t_formulas");
  std::vector<std::string>::iterator iter;
  for(iter = theExpsNames.begin(); iter!=theExpsNames.end(); iter++){
    string currName = *iter;

    if(m_dfl->getIndex(currName) != m_dfl->getNumFormulas()){
      m_tFuncsNames.push_back(currName);
    }else{
      cout<<"TFuncBase::setAttrib_DFL-----------error-------1"<<endl;
    }
  }
}


Standard_Real 
TFuncBase::
operator()(Standard_Real t) const 
{
  Standard_Real result=0.0;
  size_t nb = m_TPntVec.size();
  for(size_t i=0; i<nb; i++){
    if(t <= m_TPntVec[i]){
      m_params["t"] = t;
      m_dfl->evaluate();

      string currFuncName = m_tFuncsNames[i];
      result = (m_dfl->operator[](currFuncName)).getValue();

      break;
    }
  }
  return result;
}


/*
             t0            t1            t2            t3
-------------|-------------|-------------|-------------|
     f0             f1            f2           f3
//*/
