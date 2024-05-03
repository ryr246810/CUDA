#ifndef _FieldsAccessBase_HeaderFile
#define _FieldsAccessBase_HeaderFile

#include <FieldsBase.hxx>

class FieldsAccessBase : public FieldsBase
{

public:
  FieldsAccessBase();
  FieldsAccessBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  ~FieldsAccessBase();


public:
  inline void SetRgn(const TxSlab<Standard_Integer>& _rgn){
    m_Rgn = _rgn;
  }
  inline const TxSlab<Standard_Integer>& GetRgn() const{
    return m_Rgn;
  }

  inline void SetAccessIndex(Standard_Integer theIndex){
    m_AccessIndex = theIndex;
  };

  inline Standard_Integer GetAccessIndex(){
    return m_AccessIndex;
  };


public:
  virtual bool IsPhysDataMemoryLocated() const{
    return true;
  };


protected:
  TxSlab<Standard_Integer> m_Rgn;

  Standard_Integer m_AccessIndex;
};

#endif
