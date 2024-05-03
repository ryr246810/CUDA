#include <ComboFields_Dynamic_SweptFaceSoftMagSrc.hxx>


ComboFields_Dynamic_SweptFaceSoftMagSrc::
ComboFields_Dynamic_SweptFaceSoftMagSrc()
  :ComboFields_Dynamic_SweptFaceMagSrc()
{
}


ComboFields_Dynamic_SweptFaceSoftMagSrc::
ComboFields_Dynamic_SweptFaceSoftMagSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule)
  :ComboFields_Dynamic_SweptFaceMagSrc(theCntr, theRule)
{
}


ComboFields_Dynamic_SweptFaceSoftMagSrc::
~ComboFields_Dynamic_SweptFaceSoftMagSrc()
{
}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void ComboFields_Dynamic_SweptFaceSoftMagSrc::Setup()
{
  ComboFields_Dynamic_SweptFaceMagSrc::Setup();
  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
}

