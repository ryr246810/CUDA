#ifndef _Grid_Tool_HeaderFile
#define _Grid_Tool_HeaderFile

#include <ZRGrid_Ctrl.hxx>
#include <TxHierAttribSet.h>

// one dimension tool

class Grid_Tool
{
public:
  Grid_Tool();
  virtual ~Grid_Tool();

public:
  virtual void SetGridCtrl(ZRGrid_Ctrl* theGC);
  virtual void SetAttrib(const TxHierAttribSet& tha);
  virtual void Build();

  virtual Standard_Integer GetDir() const;

  virtual Standard_Real GetFirstStep() const;
  virtual Standard_Real GetLastStep() const;

  const vector<Standard_Real>& GetResult() const;

  vector<Standard_Real>& ModifyResult();

  Standard_Real GetOrg() const; 

protected:
  ZRGrid_Ctrl* m_GridCtrl;
  Standard_Integer m_Dir;
  vector<Standard_Real> m_CoordinateVec;
};

#endif
