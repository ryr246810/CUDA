#include <ComboFields_Dynamic_Srcs_Cyl3D.hxx>


ComboFields_Dynamic_Srcs_Cyl3D::
ComboFields_Dynamic_Srcs_Cyl3D()
  : FieldsDefineBase()
{
};


ComboFields_Dynamic_Srcs_Cyl3D::
ComboFields_Dynamic_Srcs_Cyl3D(const FieldsDefineCntr* theCntr)
  : FieldsDefineBase(theCntr)
{
};


ComboFields_Dynamic_Srcs_Cyl3D::
~ComboFields_Dynamic_Srcs_Cyl3D()
{
  for (Standard_Size idx = m_Srcs_Cyl3D.size(); idx>0; delete m_Srcs_Cyl3D[--idx]);
  m_Srcs_Cyl3D.clear();
};


void 
ComboFields_Dynamic_Srcs_Cyl3D::
Append(ComboFields_Dynamic_SrcBase* _oneNewSrc)
{
  m_Srcs_Cyl3D.push_back(_oneNewSrc);
};


void 
ComboFields_Dynamic_Srcs_Cyl3D::
Setup()
{
}

