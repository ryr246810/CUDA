#include <ComboFields_Dynamic_VertexSoftElecSrc.hxx>


ComboFields_Dynamic_VertexSoftElecSrc::
ComboFields_Dynamic_VertexSoftElecSrc()
  :ComboFields_Dynamic_VertexElecSrc()
{
}


ComboFields_Dynamic_VertexSoftElecSrc::
ComboFields_Dynamic_VertexSoftElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule)
  :ComboFields_Dynamic_VertexElecSrc(theCntr, theRule)
{
}


ComboFields_Dynamic_VertexSoftElecSrc::~ComboFields_Dynamic_VertexSoftElecSrc()
{
}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_VertexSoftElecSrc::Setup()
{
  ComboFields_Dynamic_VertexElecSrc::Setup();

  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
}

