#ifndef _EdgeElecFldsBase_HeaderFile
#define _EdgeElecFldsBase_HeaderFile

#include <FieldsBase.hxx>
#include <GridEdgeData.hxx>

class EdgeElecFldsBase : public FieldsBase
{

public:
  EdgeElecFldsBase();
  EdgeElecFldsBase(const FieldsDefineCntr* theCntr);
  EdgeElecFldsBase(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~EdgeElecFldsBase();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;


public:
  vector<GridEdgeData*>& GetDatas() {return m_Datas;};


protected:
  vector<GridEdgeData*>  m_Datas;
};

#endif
