#ifndef _FaceMagFldsBase_Cyl3D_HeaderFile
#define _FaceMagFldsBase_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class FaceMagFldsBase_Cyl3D : public FieldsBase
{

public:
  FaceMagFldsBase_Cyl3D();
  FaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr);
  FaceMagFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~FaceMagFldsBase_Cyl3D();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridFaceData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridFaceData*>  m_Datas;
};

#endif
