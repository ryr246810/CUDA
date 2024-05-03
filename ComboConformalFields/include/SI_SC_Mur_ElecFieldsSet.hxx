#ifndef _SI_SC_Mur_ElecFieldsSet_HeaderFile
#define _SI_SC_Mur_ElecFieldsSet_HeaderFile

#include <FieldsBase.hxx>
#include <GridGeometry.hxx>
#include <SI_SC_Mur_ElecFields.hxx>



class GridEdgeData;

class SI_SC_Mur_ElecFieldsSet : public FieldsBase
{
public:
  SI_SC_Mur_ElecFieldsSet();
  SI_SC_Mur_ElecFieldsSet(const FieldsDefineCntr* theCntr);

  virtual ~SI_SC_Mur_ElecFieldsSet();


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


private:
  vector<SI_SC_Mur_ElecFields*> m_MurPorts;
};

#endif
