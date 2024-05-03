#ifndef _SI_SC_MagFields_HeaderFile
#define _SI_SC_MagFields_HeaderFile

#include <FieldsBase.hxx>

#include <FaceMagFldsBase.hxx>
#include <SweptFaceMagFldsBase.hxx>


class SI_SC_MagFields : public FieldsBase
{
public:
  SI_SC_MagFields();
  SI_SC_MagFields(const FieldsDefineCntr* theCntr, 
		       PhysDataDefineRule theRule);

  virtual ~SI_SC_MagFields();


public:
  virtual void Setup();
  virtual bool IsPhysDataMemoryLocated() const;

public:
  void ZeroPhysDatas();


  //  interface function
public:
  virtual void Advance();
  virtual void Advance_SI(const Standard_Real si_scale);
  virtual void Advance_SI_Damping(const Standard_Real si_scale);


protected:
  FaceMagFldsBase* m_FaceMagFlds;
  SweptFaceMagFldsBase* m_SweptFaceMagFlds;
}; 

#endif
