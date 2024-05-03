#ifndef _DynFlds_FaceData_Dgn_HeaderFile
#define _DynFlds_FaceData_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_FaceData_Dgn : public FieldsDgnBase
{
public:
  DynFlds_FaceData_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);

  virtual ~DynFlds_FaceData_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeData();

  void InitParamt(const Standard_Size faceIndxVec[2]);


protected:
  GridFace* m_BaseFace;
  Standard_Real m_Data;
};

#endif
