#ifndef _SweptFaceMagFldsBase_Cyl3D_HeaderFile
#define _SweptFaceMagFldsBase_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class SweptFaceMagFldsBase_Cyl3D : public FieldsBase
{

public:
  SweptFaceMagFldsBase_Cyl3D();
  SweptFaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr);
  SweptFaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~SweptFaceMagFldsBase_Cyl3D();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridEdgeData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridEdgeData*>  m_Datas;
};

#endif
