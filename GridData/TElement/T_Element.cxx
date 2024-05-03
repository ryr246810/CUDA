#include <T_Element.hxx>


T_Element::T_Element()
{

}

T_Element::T_Element(DataBase* _Data, Standard_Integer _Dir)
{
  SetData(_Data);
  SetRelatedDir(_Dir);
}

T_Element::~T_Element()
{

}

void T_Element::SetData(DataBase* _Data)
{
  m_Data = _Data;
};


void 
T_Element::
SetRelatedDir(Standard_Integer aDir)
{
  m_RelatedDir = aDir;
};




DataBase* 
T_Element::
GetData() const
{
  return m_Data;
};


Standard_Integer 
T_Element::
GetRelatedDir() const 
{
  return m_RelatedDir;
};
