#include <PadeGridGenerationTool.hxx>
#include <cmath>
#include <iostream>


PadeGridGenerationTool::
~PadeGridGenerationTool()
{

}


PadeGridGenerationTool::
PadeGridGenerationTool(const Standard_Real x0, 
		       const Standard_Real length, 
		       const Standard_Integer firstResolution, 
		       const Standard_Integer lastResultion, 
		       const Standard_Real waveLength)
{
  m_X0 = x0;
  m_L = length;
  m_WaveLength = waveLength;

  Standard_Integer resolutionDiff = lastResultion - firstResolution;

  if(resolutionDiff==0){
    m_IsUniform = true;
    m_FirstResolutionRation = firstResolution;
    m_LastResolutionRation = lastResultion;

  }else{
    m_IsUniform = false;
    if( resolutionDiff<0 ){
      m_FirstResolutionRation = firstResolution;
      m_LastResolutionRation = lastResultion;
      m_IsReversed = false;
    }else{
      m_FirstResolutionRation = lastResultion;
      m_LastResolutionRation = firstResolution;
      m_IsReversed = true;
    }
  }

  m_S1 = waveLength/m_FirstResolutionRation;
  m_S2 = waveLength/m_LastResolutionRation;
}


void
PadeGridGenerationTool::
Build()
{
  m_N = ceil(m_L/sqrt(m_S1*m_S2));

  m_CoordinateVec.clear();

  if(m_IsUniform){
    Standard_Real theStep = m_L/((Standard_Real)m_N);
    m_S1 = theStep;
    m_S2 = theStep;
    for(Standard_Integer i=0; i<=m_N; i++){
      Standard_Real tmpX = m_X0 + theStep*i;
      m_CoordinateVec.push_back(tmpX);
    }
  }else{
    m_A = m_S1;
    m_C = m_S1/m_L-1.0/((Standard_Real)m_N);

    /*
    std::cout<<"m_IsReversed = "<<m_IsReversed<<std::endl;
    std::cout<<"m_S1 = "<<m_S1<<std::endl;
    std::cout<<"m_S2 = "<<m_S2<<std::endl;
    std::cout<<"m_A = "<<m_A<<std::endl;
    std::cout<<"m_C = "<<m_C<<std::endl;
    std::cout<<"m_L = "<<m_L<<std::endl;
    //*/

    if(m_IsReversed){
      for(Standard_Integer i=m_N; i>=0; i--){
	Standard_Real tmpX = m_X0 + m_L - m_A*i/(m_C*i+1.0);
	m_CoordinateVec.push_back(tmpX);
      }
    }else{
      for(Standard_Integer i=0; i<=m_N; i++){
	Standard_Real tmpX = m_X0 +  m_A*i/(m_C*i+1.0);
	m_CoordinateVec.push_back(tmpX);
      }
    }
  }
}


const std::vector<Standard_Real>& 
PadeGridGenerationTool::
GetResult() const
{
  return m_CoordinateVec;
}
