#ifndef _UniformGrid_Tool_HeaderFile
#define _UniformGrid_Tool_HeaderFile

#include <Grid_Tool.hxx>

class UniformGrid_Tool: public Grid_Tool
{
public:
  UniformGrid_Tool();
  virtual ~UniformGrid_Tool();

public:
  virtual void SetAttrib(const TxHierAttribSet& tha);
  virtual void Build();

protected:
  Standard_Integer m_WaveResolutionRatio;
  Standard_Integer m_N;
  Standard_Real m_Step;
  Standard_Real m_L;
  Standard_Real m_X0;
};

#endif
