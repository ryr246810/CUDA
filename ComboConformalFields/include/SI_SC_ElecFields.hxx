#ifndef _SI_SC_ElecFields_HeaderFile
#define _SI_SC_ElecFields_HeaderFile

#include <FieldsBase.hxx>
#include <EdgeElecFldsBase.hxx>
#include <SweptEdgeElecFldsBase.hxx>


class SI_SC_ElecFields : public FieldsBase
{
public:
  SI_SC_ElecFields();
  SI_SC_ElecFields(const FieldsDefineCntr* theCntr, 
		   PhysDataDefineRule theRule);

  virtual ~SI_SC_ElecFields();


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
  

protected:
  EdgeElecFldsBase* m_EdgeElecFlds;
  SweptEdgeElecFldsBase* m_SweptEdgeElecFlds;
};

#endif
