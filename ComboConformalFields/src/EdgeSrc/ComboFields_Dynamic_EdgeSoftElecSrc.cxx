#include <ComboFields_Dynamic_EdgeSoftElecSrc.hxx>


ComboFields_Dynamic_EdgeSoftElecSrc::ComboFields_Dynamic_EdgeSoftElecSrc()
  :ComboFields_Dynamic_EdgeElecSrc()
{
}


ComboFields_Dynamic_EdgeSoftElecSrc::ComboFields_Dynamic_EdgeSoftElecSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :ComboFields_Dynamic_EdgeElecSrc(theCntr, theRule)
{
}


ComboFields_Dynamic_EdgeSoftElecSrc::~ComboFields_Dynamic_EdgeSoftElecSrc()
{
}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_EdgeSoftElecSrc::Setup()
{
  ComboFields_Dynamic_EdgeElecSrc::Setup();

  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
}

