#ifndef _FieldsSrcBase_HeaderFile
#define _FieldsSrcBase_HeaderFile

#include <FieldsBase.hxx>

class FieldsSrcBase : public FieldsBase
{

public:
  FieldsSrcBase();
  FieldsSrcBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  ~FieldsSrcBase();


public:
  void SetRgn(const TxSlab2D<Standard_Integer>& _rgn){
    m_Rgn = _rgn;
  }

  const TxSlab2D<Standard_Integer>& GetRgn() const{
    return m_Rgn;
  }

  Standard_Integer GetJIndex(){
    return GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
  };

  Standard_Integer GetJMIndex(){
    return GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
  };

  Standard_Integer GetDynEIndex(){
    return GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  };

  Standard_Integer GetDynHIndex(){
    return GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();
  };

public:
  virtual bool IsPhysDataMemoryLocated() const{
    return true;
  };


protected:
  TxSlab2D<Standard_Integer> m_Rgn;
};

#endif
