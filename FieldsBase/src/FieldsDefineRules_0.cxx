#include <FieldsDefineRules.hxx>


void FieldsDefineRules::Set_DynamicElecField_PhysDataIndex(Standard_Integer _index)
{
  m_DynamicElecField_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_J_PhysDataIndex(Standard_Integer _index)
{
  m_J_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_PRE_PhysDataIndex(Standard_Integer _index)
{
  m_PRE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_AE_PhysDataIndex(Standard_Integer _index)
{
  m_AE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_BE_PhysDataIndex(Standard_Integer _index)
{
  m_BE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_DynamicMagField_PhysDataIndex(Standard_Integer _index)
{
  m_DynamicMagField_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_JM_PhysDataIndex(Standard_Integer _index)
{
  m_JM_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_PE1_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_PE1_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_PE2_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_PE2_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_PRE_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_PRE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_AE_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_AE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_BE_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_BE_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_CPML_PM1_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_PM1_PhysDataIndex = _index;
}

void FieldsDefineRules::Set_CPML_PM2_PhysDataIndex(Standard_Integer _index)
{
  m_CPML_PM2_PhysDataIndex = _index;
}


void FieldsDefineRules::Set_MUR_PreStep_PhysDataIndex(Standard_Integer _index)
{
  m_MUR_PreStep_PhysDataIndex = _index;
}


Standard_Integer FieldsDefineRules::GetCntrElecPhysDataNum() const
{
  return m_CntrElecPhysNum;
}


Standard_Integer FieldsDefineRules::GetCntrMagPhysDataNum() const
{
  return m_CntrMagPhysNum;
}


Standard_Integer FieldsDefineRules::Get_DynamicElecField_PhysDataIndex() const
{
  return m_DynamicElecField_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_J_PhysDataIndex() const
{
  return m_J_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_DynamicMagField_PhysDataIndex() const
{
  return m_DynamicMagField_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_JM_PhysDataIndex() const
{
  return m_JM_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_AE_PhysDataIndex() const
{
  return m_AE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_BE_PhysDataIndex() const
{
  return m_BE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_PRE_PhysDataIndex() const
{
  return m_PRE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_PE1_PhysDataIndex() const
{
  return m_CPML_PE1_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_PE2_PhysDataIndex() const
{
  return m_CPML_PE2_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_PRE_PhysDataIndex() const
{
  return m_CPML_PRE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_AE_PhysDataIndex() const
{
  return m_CPML_AE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_BE_PhysDataIndex() const
{
  return m_CPML_BE_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_PM1_PhysDataIndex() const
{
  return m_CPML_PM1_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_CPML_PM2_PhysDataIndex() const
{
  return m_CPML_PM2_PhysDataIndex;
}


Standard_Integer FieldsDefineRules::Get_MUR_PreStep_PhysDataIndex() const
{
  return m_MUR_PreStep_PhysDataIndex;
}
