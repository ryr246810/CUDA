#ifndef _DynFlds_SweptEdgeData_Dgn_HeaderFile
#define _DynFlds_SweptEdgeData_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_SweptEdgeData_Dgn : public FieldsDgnBase
{
public:
  DynFlds_SweptEdgeData_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);

  virtual ~DynFlds_SweptEdgeData_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeData();

  void InitParamt(const Standard_Integer theDir, 
		  const Standard_Size edgeIndxVec[2]);

  void broadCastData(const Standard_Real thisData);



protected:
  GridEdge* m_BaseEdge;

  // Zero the pointer until know size
  Standard_Real* m_DataOnProc;
  Standard_Real m_Data;

};

#endif
