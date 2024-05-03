#ifndef _FieldsDgnBase_HeaderFile
#define _FieldsDgnBase_HeaderFile

#include <FieldsDefineRule.hxx>
#include <FieldsDefineCntr.hxx>

#include <TxHierAttribSet.h>
#include <DynObj.hxx>

#include <string>

class FieldsDgnBase : public DynObj
{

public:
  FieldsDgnBase(){
    m_FldsDefCntr=NULL;
    m_PhiIndex = -1;
  };

  virtual void Init(const FieldsDefineCntr* theCntr) {
    m_FldsDefCntr=theCntr;
    m_PhiIndex = -1;
  };


  virtual void SetAttrib(const TxHierAttribSet& tha){};

  virtual ~FieldsDgnBase(){ };

  virtual Standard_Real GetValue() {return 0.0;};

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
  const FieldsDefineCntr* GetFldsDefCntr() const {
    return m_FldsDefCntr;
  };

protected:
  const FieldsDefineCntr* m_FldsDefCntr;
  Standard_Integer m_PhiIndex;
}; 

#endif
