#ifndef _SI_SC_CPML_MagFields_HeaderFile
#define _SI_SC_CPML_MagFields_HeaderFile

#include <SI_SC_MagFields.hxx>

class GridFaceData;
class GridEdgeData;


class SI_SC_CPML_MagFields : public SI_SC_MagFields
{
public:
  SI_SC_CPML_MagFields();
  SI_SC_CPML_MagFields(const FieldsDefineCntr* theCntr);

  virtual ~SI_SC_CPML_MagFields();


public:
  virtual void Advance();
  virtual void Advance_SI(const Standard_Real si_scale);
  virtual void Advance_SI_Damping(const Standard_Real si_scale);


public:
  virtual void Setup();
  void Setup_PML_a_b();
  void Write_PML_a_b(std::ostream& theoutstream) const;

  virtual bool IsPhysDataMemoryLocated() const;

public:
  void ZeroPhysDatas();
};

#endif
