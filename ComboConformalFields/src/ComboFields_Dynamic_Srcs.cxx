#include <ComboFields_Dynamic_Srcs.hxx>


ComboFields_Dynamic_Srcs::
ComboFields_Dynamic_Srcs()
  : FieldsDefineBase()
{
};


ComboFields_Dynamic_Srcs::
ComboFields_Dynamic_Srcs(const FieldsDefineCntr* theCntr)
  : FieldsDefineBase(theCntr)
{
};


ComboFields_Dynamic_Srcs::
~ComboFields_Dynamic_Srcs()
{
  for (Standard_Size idx = m_Srcs.size(); idx>0; delete m_Srcs[--idx]);
  m_Srcs.clear();
};


void 
ComboFields_Dynamic_Srcs::
Append(ComboFields_Dynamic_SrcBase* _oneNewSrc)
{
 m_Srcs.push_back(_oneNewSrc);
};


void 
ComboFields_Dynamic_Srcs::
Setup()
{
}

