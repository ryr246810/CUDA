#ifndef _DynFlds_Voltage_Dgn_HeaderFile
#define _DynFlds_Voltage_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_Voltage_Dgn : public FieldsDgnBase
{
public:
  DynFlds_Voltage_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);


  virtual ~DynFlds_Voltage_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeVoltage();
  void InitData();

  void broadCastVoltage(const Standard_Real thisFlux);


protected:
  Standard_Integer m_LineDir;

  TxVector2D<Standard_Size> m_OrgIndx;
  TxVector2D<Standard_Size> m_EndIndx;

  vector<GridEdgeData*> m_Datas;

  // Zero the pointer until know size
  Standard_Real m_Voltage;
};

#endif
