#ifndef _Geom_Face_TxtBuilder_HeaderFile
#define _Geom_Face_TxtBuilder_HeaderFile


#include <Geom_TxtBuilderBase.hxx>


class Geom_Face_TxtBuilder: public Geom_TxtBuilderBase
{


public:
  Geom_Face_TxtBuilder();
  ~Geom_Face_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();


public:
  Standard_Integer           m_Type;
  Handle(TDataStd_TreeNode)  m_selected_Wire_Node;
  Standard_Boolean           m_IsPlanar;
};

#endif
