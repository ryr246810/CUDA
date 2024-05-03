#ifndef _SweptEdgeElecFldsBase_Cyl3D_HeaderFile
#define _SweptEdgeElecFldsBase_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class SweptEdgeElecFldsBase_Cyl3D : public FieldsBase
{

public:
  SweptEdgeElecFldsBase_Cyl3D();
  SweptEdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr);
  SweptEdgeElecFldsBase_Cyl3D(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~SweptEdgeElecFldsBase_Cyl3D();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridVertexData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridVertexData*>  m_Datas;
};

#endif
