#ifndef _FieldsDefineBase_HeaderFile
#define _FieldsDefineBase_HeaderFile

#include <DynObj.hxx>

#include <FieldsDefineCntr.hxx>


class FieldsDefineBase: public DynObj
{

public:
  FieldsDefineBase(){
  };

  FieldsDefineBase(const FieldsDefineCntr* _cntr){
    SetFldsDefCntr(_cntr);
  };

  ~FieldsDefineBase(){
  };


public:
  inline void SetFldsDefCntr(const FieldsDefineCntr* _cntr){
    m_FldsDefCntr = _cntr;
  };


  inline const FieldsDefineCntr* GetFldsDefCntr() const {
    return m_FldsDefCntr;
  };


protected:
  const FieldsDefineCntr* m_FldsDefCntr;
};

#endif
