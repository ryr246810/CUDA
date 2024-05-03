#include <FieldsDefineCntr.hxx>


FieldsDefineCntr::FieldsDefineCntr()
{
  m_GridGeom = NULL;
  m_FieldsDataDefRules = NULL;
};


FieldsDefineCntr::~FieldsDefineCntr()
{
}


FieldsDefineCntr::
FieldsDefineCntr(const GridGeometry* _gridGeom, 
		 const FieldsDefineRules* defrules)
{
  m_GridGeom = _gridGeom;
  m_FieldsDataDefRules = defrules;
}
