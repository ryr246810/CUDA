#include <GridBndData.hxx>


Standard_Real
GridBndData::
GetEpsAccordingMatIndices( const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const
{
  Standard_Real eps = 0;
  Standard_Real nb = (Standard_Real) theMatDataIndices.size(); 
  set<Standard_Integer>::const_iterator iter;
  for(iter = theMatDataIndices.begin(); iter!=theMatDataIndices.end(); iter++){
    ISOEMMatData theData;
    if(HasMatDataWithMatIndex(*iter, theData)){
        switch(dir)
        {
        case 0: eps+=theData.m_eps_z; break;
        case 1: eps+=theData.m_eps_r; break;
        case 2: eps+=theData.m_eps_Phi; break;
	default: eps = 0; break;
        }
    }
  }
  if(nb>=1){
    eps/=nb; 
  }else{
    eps = 1.0;
  }
  return eps;
}


Standard_Real
GridBndData::
GetMuAccordingMatIndices(const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const
{
  Standard_Real mu = 0;
  Standard_Real nb = (Standard_Real) theMatDataIndices.size();
  
  set<Standard_Integer>::const_iterator iter;
  for(iter = theMatDataIndices.begin(); iter!=theMatDataIndices.end(); iter++){
    ISOEMMatData theData;
    if(HasMatDataWithMatIndex(*iter, theData)){
        switch(dir)
	{
	case 0: mu+=theData.m_mu_r; break;
	case 1: mu+=theData.m_mu_z; break;
	case 2: mu+=theData.m_mu_Phi; break;
	default: mu = 0; break;
	}
    }
  }
  if(nb>=1){
    mu/=nb; 
  }else{
    mu = 1.0;
  }
  return mu;
}


Standard_Real
GridBndData::
GetSigmaAccordingMatIndices(const set<Standard_Integer>& theMatDataIndices, Standard_Integer dir) const
{
  Standard_Real sigma = 0;
  Standard_Real nb = (Standard_Real) theMatDataIndices.size();
  
  set<Standard_Integer>::const_iterator iter;
  for(iter = theMatDataIndices.begin(); iter!=theMatDataIndices.end(); iter++){
    ISOEMMatData theData;
    if(HasMatDataWithMatIndex(*iter, theData)){
	switch(dir)
	{
	case 0: sigma+=theData.m_sigma_z; break;
	case 1: sigma+=theData.m_sigma_r; break;
	case 2: sigma+=theData.m_sigma_Phi; break;
	default: sigma = 0; break;
	}
    }
  }
  if(nb>=1){
    sigma/=nb; 
  }else{
    sigma = 0.0;
  }
  return sigma;
}
