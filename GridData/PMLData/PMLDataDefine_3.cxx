#include <PMLDataDefine.hxx>

void
PMLDataDefine::
SetAttrib( const TxHierAttribSet& inputTha)
{
  std::vector< std::string > attribNames = inputTha.getNamesOfType("PML");
  TxHierAttribSet tha = inputTha.getAttrib(attribNames[0]);

  if(tha.hasOption("key")){
    m_MethodKey = tha.getOption("key");
  }

  if(m_MethodKey==1){
    if(tha.hasOption("powerOrder")){
      m_PowOrder = tha.getOption("powerOrder");
    }
  }else{
    if(tha.hasOption("geomProgression")){
      m_GeomProg = tha.getOption("geomProgression");
    }
  }

  if(tha.hasParam("sigmaRatio")){
    m_SigmaRatio = tha.getParam("sigmaRatio");
  }
  
  if(tha.hasParam("alpha")){
    m_Alpha = tha.getParam("alpha");
  }

  if(tha.hasParam("kappaMax")){
    m_KappaMax = tha.getParam("kappaMax");
  }
}

