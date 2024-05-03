#ifndef _SweptFaceMagFldsBase_HeaderFile
#define _SweptFaceMagFldsBase_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class SweptFaceMagFldsBase : public FieldsBase
{

public:
  SweptFaceMagFldsBase();
  SweptFaceMagFldsBase(const FieldsDefineCntr* theCntr);
  SweptFaceMagFldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~SweptFaceMagFldsBase();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridEdgeData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridEdgeData*>  m_Datas;
};

#endif
