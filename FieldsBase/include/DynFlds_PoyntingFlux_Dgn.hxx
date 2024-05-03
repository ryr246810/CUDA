#ifndef _DynFlds_PoyntingFlux_Dgn_HeaderFile
#define _DynFlds_PoyntingFlux_Dgn_HeaderFile

#include <FieldsDgnBase.hxx>

class DynFlds_PoyntingFlux_Dgn : public FieldsDgnBase
{
public:
  DynFlds_PoyntingFlux_Dgn();

  virtual void Init(const FieldsDefineCntr* theCntr);

  virtual void SetAttrib(const TxHierAttribSet& tha);


  virtual ~DynFlds_PoyntingFlux_Dgn();


public:
  virtual Standard_Real GetValue();
  virtual void Advance();

private:
  void ComputeTotalPoyntingFlux();
  void InitParamt(const Standard_Integer theDir, 
		  const TxSlab<Standard_Integer>& theRgn);

  void broadCastData(const Standard_Real thisData);

  void AddFluxElement(const Standard_Size globalIndxVec[2],
		      const int edgePhysDataIndex, 
		      const int facePhysDataIndex);


protected:
  Standard_Integer m_Dir;
  TxSlab2D<Standard_Integer> m_Rgn;

  Standard_Real m_poynFlux;
};

#endif
