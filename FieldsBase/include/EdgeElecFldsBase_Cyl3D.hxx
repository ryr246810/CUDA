#ifndef _EdgeElecFldsBase_Cyl3D_HeaderFile
#define _EdgeElecFldsBase_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class EdgeElecFldsBase_Cyl3D : public FieldsBase
{

public:
  EdgeElecFldsBase_Cyl3D();
  EdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr);
  EdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~EdgeElecFldsBase_Cyl3D();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridEdgeData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridEdgeData*>  m_Datas;
};

#endif
