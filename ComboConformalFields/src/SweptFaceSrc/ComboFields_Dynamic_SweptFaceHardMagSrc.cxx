#include <ComboFields_Dynamic_SweptFaceHardMagSrc.hxx>


ComboFields_Dynamic_SweptFaceHardMagSrc::
ComboFields_Dynamic_SweptFaceHardMagSrc()
  :ComboFields_Dynamic_SweptFaceMagSrc()
{

}



ComboFields_Dynamic_SweptFaceHardMagSrc::
ComboFields_Dynamic_SweptFaceHardMagSrc(const FieldsDefineCntr* theCntr,
					PhysDataDefineRule theRule)
  :ComboFields_Dynamic_SweptFaceMagSrc(theCntr, theRule)
{

}


ComboFields_Dynamic_SweptFaceHardMagSrc::
~ComboFields_Dynamic_SweptFaceHardMagSrc()
{

}


// should do computing in current GeomGeometry Region define
// this function should be modified to be suitable for MPI
void 
ComboFields_Dynamic_SweptFaceHardMagSrc::
Setup()
{
  ComboFields_Dynamic_SweptFaceMagSrc::Setup();
  m_PhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
}
