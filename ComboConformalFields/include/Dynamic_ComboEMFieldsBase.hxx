#ifndef _Dynamic_ComboEMFieldsBase_HeaderFile
#define _Dynamic_ComboEMFieldsBase_HeaderFile

#include <FieldsDefineBase.hxx>
#include <FieldsDefineCntr.hxx>
#include <DynObj.hxx>
#include <TxHierAttribSet.h>
#include <string>

class ComboFields_Dynamic_Srcs;
class ComboFields_Dynamic_Srcs_Cyl3D;

class Dynamic_ComboEMFieldsBase: public FieldsDefineBase
{
public:
  Dynamic_ComboEMFieldsBase();
  Dynamic_ComboEMFieldsBase(const FieldsDefineCntr* _cntr); 
  virtual ~Dynamic_ComboEMFieldsBase();
  

public:
  virtual void Advance() = 0;
  virtual void AdvanceWithDamping() = 0;
  virtual void BuildCUDADatas() = 0;
  virtual void CleanCUDADatas() = 0;
  virtual void TransDgnData() = 0;
  virtual void InitMatrixData() = 0;

  virtual vector<GridEdgeData*>& GetCntrElecEdges() = 0;// added 2019.4/15
  virtual vector<GridVertexData*>& GetCntrElecVertices() = 0;// added 2019.4/15
  
  
public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() = 0;
  virtual void ZeroPhysDatas() = 0;

  virtual void SetOrder(const Standard_Integer theOrder) {};
  virtual void SetDamping(const Standard_Real theDamping) {};

public:
  virtual void Write_PML_Inf() = 0;
  virtual void Get_Ezr_Info(Standard_Real** ptr, Standard_Size* size){};
  virtual void Get_Ephi_Info(Standard_Real** ptr, Standard_Size* size){};
  virtual void Get_Mzr_Info(Standard_Real** ptr, Standard_Size* size){};
  virtual void Get_Mphi_Info(Standard_Real** ptr, Standard_Size* size){};

  virtual void Get_cuda_ptr(Standard_Real** EzrPtr, Standard_Real** EphiPtr, Standard_Real** MzrPtr, Standard_Real** MphiPtr){};

public:
  virtual void InitFldSrcs();
  virtual void InitFldSrcs_Cyl3D();
  virtual void SetFldSrcsAttrib(const std::string& theWorkDir,
				const TxHierAttribSet& theFaceBndTha);
  virtual void SetFldSrcsAttrib_Cyl3D(const std::string& theWorkDir,
				const TxHierAttribSet& theFaceBndTha);

protected:
  ComboFields_Dynamic_Srcs* m_FldSrcs;
  ComboFields_Dynamic_Srcs_Cyl3D* m_FldSrcs_Cyl3D;
};

#endif

