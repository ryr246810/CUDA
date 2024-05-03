#ifndef _Geom_Revolution_TxtBuilder_HeaderFile
#define _Geom_Revolution_TxtBuilder_HeaderFile


#include <Geom_TxtBuilderBase.hxx>


class Geom_Revolution_TxtBuilder: public Geom_TxtBuilderBase
{
public:
  Geom_Revolution_TxtBuilder();
  ~Geom_Revolution_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();

public:
  Handle(TDataStd_TreeNode)  m_BaseNode;
  Handle(TDataStd_TreeNode)  m_AxisNode;
  Standard_Real             m_Angle;
  Standard_Integer          m_Type;
};

#endif
