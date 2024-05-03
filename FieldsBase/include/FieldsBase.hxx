#ifndef _FieldsBase_HeaderFile
#define _FieldsBase_HeaderFile

#include <FieldsDefineRule.hxx>
#include <FieldsDefineCntr.hxx>

#include <DynObj.hxx>

#include <set>
#include <vector>
using namespace std;


class FieldsBase : public DynObj
{

public:
  FieldsBase(){
    // m_FldsDefCntr==NULL;
    m_Rule = NORULE;
  };


  FieldsBase(const FieldsDefineCntr* theCntr){
    Init(theCntr, NORULE);
  };


  FieldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule){
    Init(theCntr, theRule);
  };


  virtual ~FieldsBase(){
    m_Materials.clear();
  };


public:
  void Init(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule) {
    m_FldsDefCntr=theCntr;
    m_Rule = theRule;
  };

  virtual void Setup(){};

public:
  inline void SetMaterials(const set<Standard_Integer>& theMaterials){
    m_Materials = theMaterials;
  }

  inline void AppendingMaterial(Standard_Integer theMaterial){
    m_Materials.insert(theMaterial);
  };

  inline void EraseMaterial(Standard_Integer theMaterial){
    set<Standard_Integer>::iterator iter = m_Materials.find(theMaterial);
    if(iter!=m_Materials.end()){
      m_Materials.erase(iter);
    }
  };

  inline void WriteMaterial(){
    cout<<"FieldsBase::WriteMaterial==============>>"<<endl;
    set<Standard_Integer>::iterator iter;
    for(iter = m_Materials.begin(); iter != m_Materials.end(); iter++){
      cout<<*iter<<endl;
    }
    cout<<"FieldsBase::WriteMaterial==============<<"<<endl;
  };

  inline void clearMaterials(){
    m_Materials.clear();
  };


public:
  virtual bool IsPhysDataMemoryLocated() const = 0;

public:
  const GridGeometry* GetGridGeom() const {
    return m_FldsDefCntr->GetGridGeom();
  };

  const GridGeometry_Cyl3D* GetGridGeom_Cyl3D() const {
    return m_FldsDefCntr->GetGridGeom_Cyl3D();
  };

  const GridGeometry* GetGridGeom(Standard_Integer i) const {
    if(i == -1) return m_FldsDefCntr->GetGridGeom();
    else return m_FldsDefCntr->GetGridGeom_Cyl3D()->GetGridGeometry(i);
  };

  const ZRGrid* GetZRGrid() const
  {
    return m_FldsDefCntr->GetZRGrid();
  }

  const FieldsDefineCntr* GetFldsDefCntr() const {
    return m_FldsDefCntr;
  };

  const GridBndData* GetGridBndDatas() const{
    return m_FldsDefCntr->GetGridBndDatas();
  }

protected:
  const FieldsDefineCntr* m_FldsDefCntr;

  PhysDataDefineRule m_Rule;

  set<Standard_Integer> m_Materials;
}; 

#endif
