#ifndef _SI_SC_ComboEMFields_Cyl3D_HeaderFile
#define _SI_SC_ComboEMFields_Cyl3D_HeaderFile

#include <Dynamic_ComboEMFieldsBase.hxx>

#include <SI_SC_ElecFields_Cyl3D.hxx>
#include <SI_SC_MagFields_Cyl3D.hxx>

#include <SI_SC_CPML_ElecFields_Cyl3D.hxx>
#include <SI_SC_CPML_MagFields_Cyl3D.hxx>
#include <SI_SC_Mur_ElecFieldsSet_Cyl3D.hxx>

#include <SI_SC_ElecFields_Cyl3D.hxx>
#include <SI_SC_MagFields_Cyl3D.hxx>


class SI_SC_ComboEMFields_Cyl3D: public Dynamic_ComboEMFieldsBase
{
 public:
  SI_SC_ComboEMFields_Cyl3D();
  SI_SC_ComboEMFields_Cyl3D(const FieldsDefineCntr* _cntr); 
  virtual ~SI_SC_ComboEMFields_Cyl3D();

public:
  virtual void Advance();
  virtual void AdvanceWithDamping();
  virtual void BuildCUDADatas();
  virtual void CleanCUDADatas();
  virtual void TransDgnData();
  virtual void InitMatrixData();

public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated();
  virtual void ZeroPhysDatas();

  virtual void SetOrder(const Standard_Integer theOrder);
  virtual void SetDamping(const Standard_Real theDamping);

  virtual vector<GridEdgeData*>& GetCntrElecEdges();// added 2019.4/15
  virtual vector<GridVertexData*>& GetCntrElecVertices();// added 2019.4/15
  
public:
  virtual void Write_PML_Inf();


  // Interface function
private:
  void SetupCntrElecFields();
  void SetupCntrMagFields();


  void SetupCPMLElecFields();
  void SetupCPMLMagFields();
  void SetupMurElecFields();


private:
  Standard_Integer m_Order;
  Standard_Integer m_Step;

  Standard_Real* m_b;
  Standard_Real* m_bb;

  Standard_Real m_Damping;


  SI_SC_ElecFields_Cyl3D* m_ECntrFields_Cyl3D;
  SI_SC_MagFields_Cyl3D*  m_MCntrFields_Cyl3D;

  SI_SC_CPML_ElecFields_Cyl3D* m_ECPMLFields_Cyl3D;
  SI_SC_CPML_MagFields_Cyl3D*  m_MCPMLFields_Cyl3D;
  SI_SC_Mur_ElecFieldsSet_Cyl3D* m_EMurFields_Cyl3D;



};

#endif
