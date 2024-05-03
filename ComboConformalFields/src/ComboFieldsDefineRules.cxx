#include <ComboFieldsDefineRules.hxx>
#include <BaseDataDefine.hxx>

ComboFieldsDefineRules::ComboFieldsDefineRules()
 : FieldsDefineRules()
{

}


ComboFieldsDefineRules::~ComboFieldsDefineRules()
{
}



void ComboFieldsDefineRules::Setup_Fields_PhysDatasNum_AccordingMaterialDefine()
{
  SetCntrElecPhysDataNum(5);  // dynamic(0), current(1), AE(2), BE(3), PRE(4), here AE and BE is for damping
  Set_DynamicElecField_PhysDataIndex(0);
  Set_J_PhysDataIndex(1);
  Set_AE_PhysDataIndex(2);
  Set_BE_PhysDataIndex(3);
  Set_PRE_PhysDataIndex(4);


  SetCntrMagPhysDataNum(2);  // dynamic(0), current(1)
  Set_DynamicMagField_PhysDataIndex(0);
  Set_JM_PhysDataIndex(1);


  SetBndElecPhysDataNum(Standard_Integer(PML), 7);   // pml: dynamic(0), current(1), AE(2), BE(3), PRE(4), PE1(5), PE2(6), here AE and BE is for damping
  Set_CPML_AE_PhysDataIndex(2);
  Set_CPML_BE_PhysDataIndex(3);
  Set_CPML_PRE_PhysDataIndex(4);
  Set_CPML_PE1_PhysDataIndex(5);
  Set_CPML_PE2_PhysDataIndex(6);


  SetBndElecPhysDataNum(Standard_Integer(MUR), 6);   // mur dynamic(0), current(1), AE(2), BE(3), prestep(4), here AE and BE is for damping
  Set_MUR_PreStep_PhysDataIndex(5);

  SetBndMagPhysDataNum(Standard_Integer(MUR), 2);   // mur dynamic, current

  SetBndMagPhysDataNum(Standard_Integer(PML), 4);   // pml dynamic, current, PM1, PM2
  Set_CPML_PM1_PhysDataIndex(2);
  Set_CPML_PM2_PhysDataIndex(3);

  SetBndElecPhysDataNum(Standard_Integer(PEC), 2);   // dynamic, current
  SetBndMagPhysDataNum(Standard_Integer(PEC), 2);   // dynamic, current

  /*
  SetBndElecPhysDataNum(Standard_Integer(PEC), 0); 
  SetBndMagPhysDataNum(Standard_Integer(PEC), 0); 
  //*/
}
