#ifndef _Geom_Vector_TxtBuilder_HeaderFile
#define _Geom_Vector_TxtBuilder_HeaderFile


#include <Geom_TxtBuilderBase.hxx>


class Geom_Vector_TxtBuilder: public Geom_TxtBuilderBase
{
public:
  Geom_Vector_TxtBuilder();
  ~Geom_Vector_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();

public:
  Standard_Integer m_Type;
  double  m_Dim[3];

public:
  Handle(TDataStd_TreeNode)  m_selected_Vertex1_Node;
  Handle(TDataStd_TreeNode)  m_selected_Vertex2_Node;
};

#endif
