#include <ComboFields_Dynamic_VertexHardElecSrc.hxx>


ComboFields_Dynamic_VertexHardElecSrc::
ComboFields_Dynamic_VertexHardElecSrc()
  :ComboFields_Dynamic_VertexElecSrc()
{

}



ComboFields_Dynamic_VertexHardElecSrc::
ComboFields_Dynamic_VertexHardElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule)
  :ComboFields_Dynamic_VertexElecSrc(theCntr, theRule)
{

}


ComboFields_Dynamic_VertexHardElecSrc::~ComboFields_Dynamic_VertexHardElecSrc()
{

}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_VertexHardElecSrc::Setup()
{
  ComboFields_Dynamic_VertexElecSrc::Setup();
  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  m_PhysJDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
}
