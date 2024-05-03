#ifndef _SI_SC_CPML_ElecFields_HeaderFile
#define _SI_SC_CPML_ElecFields_HeaderFile

#include <SI_SC_ElecFields.hxx>


class SI_SC_CPML_ElecFields : public SI_SC_ElecFields
{
public:
  SI_SC_CPML_ElecFields();
  SI_SC_CPML_ElecFields(const FieldsDefineCntr* theCntr);
  virtual ~SI_SC_CPML_ElecFields();
  
public:
  virtual void Advance();
  virtual void Advance_SI(const Standard_Real si_scale);
  virtual void Advance_SI_Damping(const Standard_Real si_scale,
				  const Standard_Real damping_scale);

public:
  virtual void Setup();
  void Setup_PML_a_b();
  void Write_PML_a_b(std::ostream& theoutstream) const;
  virtual bool IsPhysDataMemoryLocated() const;

public:
  void ZeroPhysDatas();
};

#endif
