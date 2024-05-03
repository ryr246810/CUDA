#ifndef _PadeGridGenerationTool_HeaderFile
#define _PadeGridGenerationTool_HeaderFile

#include <TxHierAttribSet.h>
#include <Standard_TypeDefine.hxx>
#include <vector>

class PadeGridGenerationTool
{
public:
  PadeGridGenerationTool(const Standard_Real x0, 
			 const Standard_Real length, 
			 const Standard_Integer firstResolution, 
			 const Standard_Integer lastResultion, 
			 const Standard_Real waveLength);
  
  ~PadeGridGenerationTool();
  
public:
  void Build();
  const std::vector<Standard_Real>& GetResult() const;
  
private:
  Standard_Real m_X0;
  Standard_Real m_L;
  
  Standard_Integer m_FirstResolutionRation;
  Standard_Integer m_LastResolutionRation;
  
  Standard_Real m_WaveLength;
  
private:
  bool m_IsUniform;
  bool m_IsReversed;
  Standard_Real m_S1;
  Standard_Real m_S2;
  
  Standard_Real m_A;
  Standard_Real m_C;
  Standard_Integer m_N;
  
private:
  std::vector<Standard_Real> m_CoordinateVec;
  
private:
  PadeGridGenerationTool(){};
};

#endif
