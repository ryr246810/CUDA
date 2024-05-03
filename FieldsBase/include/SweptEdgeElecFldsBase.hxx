#ifndef _SweptEdgeElecFldsBase_HeaderFile
#define _SweptEdgeElecFldsBase_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class SweptEdgeElecFldsBase : public FieldsBase
{

public:
  SweptEdgeElecFldsBase();
  SweptEdgeElecFldsBase(const FieldsDefineCntr* theCntr);
  SweptEdgeElecFldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~SweptEdgeElecFldsBase();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridVertexData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridVertexData*>  m_Datas;
};

#endif
