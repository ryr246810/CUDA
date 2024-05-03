#ifndef _Geom_Cylinder_TxtBuilder_HeaderFile
#define _Geom_Cylinder_TxtBuilder_HeaderFile


#include <Geom_TxtBuilderBase.hxx>


class Geom_Cylinder_TxtBuilder: public Geom_TxtBuilderBase
{
public:
  Geom_Cylinder_TxtBuilder();
  ~Geom_Cylinder_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();

public:
  Standard_Integer          m_Type;
  double  m_R;
  double  m_H;

public:
  double  m_Org[3];
  double  m_Dir[3];

public:
  Handle(TDataStd_TreeNode)  m_selected_Vertex_Node;
  Handle(TDataStd_TreeNode)  m_selected_Vector_Node;
};

#endif
