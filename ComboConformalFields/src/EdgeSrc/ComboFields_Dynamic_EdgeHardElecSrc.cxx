#include <ComboFields_Dynamic_EdgeHardElecSrc.hxx>


ComboFields_Dynamic_EdgeHardElecSrc::ComboFields_Dynamic_EdgeHardElecSrc()
  :ComboFields_Dynamic_EdgeElecSrc()
{

}



ComboFields_Dynamic_EdgeHardElecSrc::ComboFields_Dynamic_EdgeHardElecSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :ComboFields_Dynamic_EdgeElecSrc(theCntr, theRule)
{

}


ComboFields_Dynamic_EdgeHardElecSrc::~ComboFields_Dynamic_EdgeHardElecSrc()
{

}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_EdgeHardElecSrc::Setup()
{
  ComboFields_Dynamic_EdgeElecSrc::Setup();
  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
}
