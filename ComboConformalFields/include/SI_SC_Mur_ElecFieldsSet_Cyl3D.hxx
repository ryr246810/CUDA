#ifndef _SI_SC_Mur_ElecFieldsSet_Cyl3D_HeaderFile
#define _SI_SC_Mur_ElecFieldsSet_Cyl3D_HeaderFile

#include <FieldsBase.hxx>
#include <GridGeometry.hxx>
#include <SI_SC_Mur_ElecFields.hxx>



class GridEdgeData;

class SI_SC_Mur_ElecFieldsSet_Cyl3D : public FieldsBase
{
public:
  SI_SC_Mur_ElecFieldsSet_Cyl3D();
  SI_SC_Mur_ElecFieldsSet_Cyl3D(const FieldsDefineCntr* theCntr);

  virtual ~SI_SC_Mur_ElecFieldsSet_Cyl3D();


public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const {return true;};

public:
  void ZeroPhysDatas();
  void Clear();

public:
  virtual void Advance_SI(const Standard_Real si_scale);
  virtual void Advance_SI_Damping(const Standard_Real si_scale, 
				  const Standard_Real damping_scale);
  virtual void Advance();

public:
  SI_SC_Mur_ElecFields* Get_subDatas(int idx){return m_MurPorts_Cyl3D[idx];};
  int Get_MurPortsDatasNum(){return m_MurPorts_Cyl3D.size();};

private:
  vector<SI_SC_Mur_ElecFields*> m_MurPorts_Cyl3D;
};

#endif
