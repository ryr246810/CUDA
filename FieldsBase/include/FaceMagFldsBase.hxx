#ifndef _FaceMagFldsBase_HeaderFile
#define _FaceMagFldsBase_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class FaceMagFldsBase : public FieldsBase
{

public:
  FaceMagFldsBase();
  FaceMagFldsBase(const FieldsDefineCntr* theCntr);
  FaceMagFldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~FaceMagFldsBase();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridFaceData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridFaceData*>  m_Datas;
};

#endif
