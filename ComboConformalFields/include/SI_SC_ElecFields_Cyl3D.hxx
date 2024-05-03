#ifndef _SI_SC_ElecFields_Cyl3D_HeaderFile
#define _SI_SC_ElecFields_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <EdgeElecFldsBase_Cyl3D.hxx>
#include <SweptEdgeElecFldsBase_Cyl3D.hxx>
#include <EdgeElecFldsBase_Axis_Cyl3D.hxx>


class SI_SC_ElecFields_Cyl3D : public FieldsBase
{
public:
  SI_SC_ElecFields_Cyl3D();
  SI_SC_ElecFields_Cyl3D(const FieldsDefineCntr* theCntr, 
		   PhysDataDefineRule theRule);

  virtual ~SI_SC_ElecFields_Cyl3D();


public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;

public:
  void ZeroPhysDatas();
  void SetupGridEdgeDatasEfficientLength();

public:
  virtual void Advance();
  virtual void Advance_SI(const Standard_Real si_scale);
  virtual void Advance_SI_Damping(const Standard_Real si_scale, 
				  const Standard_Real damping_scale);
  
  vector<GridEdgeData*>& GetEdgeDatas();// added 2019.4/15
  vector<GridVertexData*>& GetVertexDatas();// added 2019.4/15
  

public:
  EdgeElecFldsBase_Cyl3D* m_EdgeElecFlds_Cyl3D;
  SweptEdgeElecFldsBase_Cyl3D* m_SweptEdgeElecFlds_Cyl3D;
  EdgeElecFldsBase_Axis_Cyl3D* m_EdgeElecFlds_Axis_Cyl3D;
};

#endif
