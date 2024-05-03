#ifndef _DynFlds_VertexData_Dgn_HeaderFile
#define _DynFlds_VertexData_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_VertexData_Dgn : public FieldsDgnBase
{
public:
  DynFlds_VertexData_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);

  virtual ~DynFlds_VertexData_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeData();

  void InitParamt(const Standard_Size edgeIndxVec[2]);

  void broadCastData(const Standard_Real thisData);



protected:
  GridVertexData* m_Vertex;

  // Zero the pointer until know size
  Standard_Real* m_DataOnProc;
  Standard_Real m_Data;

};

#endif
