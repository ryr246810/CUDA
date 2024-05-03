#ifndef _PadeGrid_Tool_HeaderFile
#define _PadeGrid_Tool_HeaderFile

#include <Grid_Tool.hxx>
#include <PadeGridGenerationTool.hxx>

class PadeGrid_Tool: public Grid_Tool
{
  typedef struct{
    Standard_Real m_Coord;
    Standard_Integer m_Step;
  } PadePnt;

public:
  PadeGrid_Tool();
  virtual ~PadeGrid_Tool();

public:
  virtual void SetAttrib(const TxHierAttribSet& tha);
  virtual void Build();

private:
  void SetupGridGenerationTool();
  void ClearTools();

protected:
  vector<PadePnt> m_PadePntVec;
  vector<PadeGridGenerationTool*> m_ToolVec;
};

#endif
