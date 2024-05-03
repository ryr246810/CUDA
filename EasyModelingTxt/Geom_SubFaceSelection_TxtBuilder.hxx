//2010.05.07
//wang yue
//id_wangyue@hotmail.com

#ifndef _Geom_SubFaceSelection_TxtBuilder_HeaderFile
#define _Geom_SubFaceSelection_TxtBuilder_HeaderFile

#include <Geom_TxtBuilderBase.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopoDS_Shape.hxx>

class Geom_SubFaceSelection_TxtBuilder : public Geom_TxtBuilderBase
{
public:
  Geom_SubFaceSelection_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();


protected:
  Handle(TDataStd_TreeNode)  m_ContextNode;

  double  m_RefPnt[3];

  TopAbs_ShapeEnum          m_SelectionShapeType;

  TopoDS_Shape              m_SelectedShape;
  Standard_Boolean          m_IsDone;
};


#endif
