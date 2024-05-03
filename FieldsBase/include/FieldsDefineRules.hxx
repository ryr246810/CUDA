#ifndef _FieldsDefineRules_HeaderFile
#define _FieldsDefineRules_HeaderFile

#include <Standard_TypeDefine.hxx>
#include <set>
#include <vector>
#include <map>

class FieldsDefineRules
{
public:
  FieldsDefineRules();
  ~FieldsDefineRules();

  void ClearPhysDataNumDefine()
  {
    m_BndElecMaterialPhysDataNumMap.clear();
    m_BndMagMaterialPhysDataNumMap.clear();
  }


public:
  virtual void Setup_Fields_PhysDatasNum_AccordingMaterialDefine(){};


public:
  Standard_Integer GetCntrElecPhysDataNum() const;
  Standard_Integer GetCntrMagPhysDataNum() const;

  Standard_Integer GetBndElecPhysDataNum(const Standard_Integer _material) const;
  Standard_Integer GetBndMagPhysDataNum(const Standard_Integer _material) const;

  void GetBndElecMaterialSet(std::set<Standard_Integer> & dataSet) const;
  void GetBndMagMaterialSet(std::set<Standard_Integer> & dataSet) const;

  bool IsBndElecMaterial(const Standard_Integer _material) const;
  bool IsBndMagMaterial(const Standard_Integer _material) const;


public:
  virtual Standard_Integer Get_DynamicElecField_PhysDataIndex() const;
  virtual Standard_Integer Get_J_PhysDataIndex() const;
  virtual Standard_Integer Get_AE_PhysDataIndex() const;
  virtual Standard_Integer Get_BE_PhysDataIndex() const;
  virtual Standard_Integer Get_PRE_PhysDataIndex() const;

  virtual Standard_Integer Get_DynamicMagField_PhysDataIndex() const;
  virtual Standard_Integer Get_JM_PhysDataIndex() const;


  virtual Standard_Integer Get_CPML_AE_PhysDataIndex() const;
  virtual Standard_Integer Get_CPML_BE_PhysDataIndex() const;

  virtual Standard_Integer Get_CPML_PE1_PhysDataIndex() const;
  virtual Standard_Integer Get_CPML_PE2_PhysDataIndex() const;
  virtual Standard_Integer Get_CPML_PRE_PhysDataIndex() const;


  virtual Standard_Integer Get_CPML_PM1_PhysDataIndex() const;
  virtual Standard_Integer Get_CPML_PM2_PhysDataIndex() const;

  virtual Standard_Integer Get_MUR_PreStep_PhysDataIndex() const;

public:
  virtual void Set_DynamicElecField_PhysDataIndex(Standard_Integer _index);
  virtual void Set_J_PhysDataIndex(Standard_Integer _index);
  virtual void Set_AE_PhysDataIndex(Standard_Integer _index);
  virtual void Set_BE_PhysDataIndex(Standard_Integer _index);
  virtual void Set_PRE_PhysDataIndex(Standard_Integer _index);

  virtual void Set_DynamicMagField_PhysDataIndex(Standard_Integer _index);
  virtual void Set_JM_PhysDataIndex(Standard_Integer _index);


  virtual void Set_CPML_AE_PhysDataIndex(Standard_Integer _index);
  virtual void Set_CPML_BE_PhysDataIndex(Standard_Integer _index);
  virtual void Set_CPML_PRE_PhysDataIndex(Standard_Integer _index);

  virtual void Set_CPML_PE1_PhysDataIndex(Standard_Integer _index);
  virtual void Set_CPML_PE2_PhysDataIndex(Standard_Integer _index);

  virtual void Set_CPML_PM1_PhysDataIndex(Standard_Integer _index);
  virtual void Set_CPML_PM2_PhysDataIndex(Standard_Integer _index);

  virtual void Set_MUR_PreStep_PhysDataIndex(Standard_Integer _index);


  void SetBndElecPhysDataNum(Standard_Integer _material, Standard_Integer _edgePhysDataNum);
  void SetBndMagPhysDataNum(Standard_Integer _material, Standard_Integer _facePhysDataNum);

  void SetCntrElecPhysDataNum(Standard_Integer _elecPhysDataNum){m_CntrElecPhysNum =_elecPhysDataNum;};
  void SetCntrMagPhysDataNum(Standard_Integer _magPhysDataNum){m_CntrMagPhysNum =_magPhysDataNum;};



private:
  Standard_Integer m_CntrElecPhysNum;
  Standard_Integer m_CntrMagPhysNum;


  Standard_Integer m_DynamicElecField_PhysDataIndex;
  Standard_Integer m_J_PhysDataIndex;
  Standard_Integer m_AE_PhysDataIndex;
  Standard_Integer m_BE_PhysDataIndex;
  Standard_Integer m_PRE_PhysDataIndex;


  Standard_Integer m_DynamicMagField_PhysDataIndex;
  Standard_Integer m_JM_PhysDataIndex;


  Standard_Integer m_CPML_AE_PhysDataIndex;
  Standard_Integer m_CPML_BE_PhysDataIndex;
  Standard_Integer m_CPML_PRE_PhysDataIndex;


  Standard_Integer m_CPML_PE1_PhysDataIndex;
  Standard_Integer m_CPML_PE2_PhysDataIndex;

  Standard_Integer m_CPML_PM1_PhysDataIndex;
  Standard_Integer m_CPML_PM2_PhysDataIndex;


  Standard_Integer m_MUR_PreStep_PhysDataIndex;


  std::map<Standard_Integer, Standard_Integer> m_BndElecMaterialPhysDataNumMap;
  std::map<Standard_Integer, Standard_Integer> m_BndMagMaterialPhysDataNumMap;
};

#endif
