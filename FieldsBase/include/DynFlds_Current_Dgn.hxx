#ifndef _DynFlds_Current_Dgn_HeaderFile
#define _DynFlds_Current_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_Current_Dgn : public FieldsDgnBase
{
public:
  DynFlds_Current_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);


  virtual ~DynFlds_Current_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeTotalCurrent();
  void InitParamt(const Standard_Integer theDir, 
		  const Standard_Integer theCutIndex);

  void broadCastCurrent(const Standard_Real thisFlux);

  void AddFluxElement(const Standard_Size globalIndxVec[3],
		      const int edgePhysDataIndex, 
		      const int facePhysDataIndex);


protected:
  Standard_Integer m_Dir;
  TxSlab2D<Standard_Integer> m_Rgn;

  // Zero the pointer until know size
  Standard_Real m_Current;
};

#endif
