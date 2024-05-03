#include <FieldsDefineRules.hxx>
#include <BaseDataDefine.hxx>


void 
FieldsDefineRules::
SetBndElecPhysDataNum(Standard_Integer _material, 
		      Standard_Integer _elecPhysDataNum)
{
  std::map<Standard_Integer, Standard_Integer>::iterator iter = m_BndElecMaterialPhysDataNumMap.find(_material);
  if(iter==m_BndElecMaterialPhysDataNumMap.end()){
    m_BndElecMaterialPhysDataNumMap.insert(std::pair<Standard_Integer, Standard_Integer>(_material, _elecPhysDataNum));
  }else{
    iter->second = _elecPhysDataNum;
  }
}


void 
FieldsDefineRules::
SetBndMagPhysDataNum(Standard_Integer _material, 
		     Standard_Integer _magPhysDataNum)
{
  std::map<Standard_Integer, Standard_Integer>::iterator iter = m_BndMagMaterialPhysDataNumMap.find(_material);
  if(iter==m_BndMagMaterialPhysDataNumMap.end()){
    m_BndMagMaterialPhysDataNumMap.insert(std::pair<Standard_Integer, Standard_Integer>(_material, _magPhysDataNum));
  }else{
    iter->second = _magPhysDataNum;
  }
}


Standard_Integer 
FieldsDefineRules::
GetBndElecPhysDataNum(const Standard_Integer _material) const
{
  Standard_Integer result = m_CntrElecPhysNum;
  std::map<Standard_Integer, Standard_Integer>::const_iterator iter = m_BndElecMaterialPhysDataNumMap.find(_material);
  if(iter!=m_BndElecMaterialPhysDataNumMap.end()) result = iter->second ;
  return result;
}


Standard_Integer 
FieldsDefineRules::
GetBndMagPhysDataNum(const Standard_Integer _material) const
{
  Standard_Integer result = m_CntrMagPhysNum;
  std::map<Standard_Integer, Standard_Integer>::const_iterator iter = m_BndMagMaterialPhysDataNumMap.find(_material);
  if(iter!=m_BndMagMaterialPhysDataNumMap.end()) result = iter->second ;
  return result;
}


void 
FieldsDefineRules::
GetBndElecMaterialSet(std::set<Standard_Integer> & dataSet) const
{
  dataSet.clear();
  std::map<Standard_Integer, Standard_Integer>::const_iterator iter;
  for( iter = m_BndElecMaterialPhysDataNumMap.begin();  iter!=m_BndElecMaterialPhysDataNumMap.end(); iter++){
    dataSet.insert(iter->first);
  }
}


void 
FieldsDefineRules::
GetBndMagMaterialSet(std::set<Standard_Integer> & dataSet) const
{
  dataSet.clear();

  std::map<Standard_Integer, Standard_Integer>::const_iterator iter;
  for( iter = m_BndMagMaterialPhysDataNumMap.begin();  iter!=m_BndMagMaterialPhysDataNumMap.end(); iter++){
    dataSet.insert(iter->first);
  }
}


bool 
FieldsDefineRules::
IsBndElecMaterial(const Standard_Integer _material) const
{
  bool result = false;
  std::map<Standard_Integer, Standard_Integer>::const_iterator iter = m_BndElecMaterialPhysDataNumMap.find(_material);
  if(iter!=m_BndElecMaterialPhysDataNumMap.end()) result = true;
  return result;
}


bool 
FieldsDefineRules::
IsBndMagMaterial(const Standard_Integer _material) const
{
  bool result = false;
  std::map<Standard_Integer, Standard_Integer>::const_iterator iter = m_BndMagMaterialPhysDataNumMap.find(_material);
  if(iter!=m_BndMagMaterialPhysDataNumMap.end()) result = true;
  return result;
}
