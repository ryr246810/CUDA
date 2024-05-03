#ifndef _SI_SC_MagFields_Cyl3D_HeaderFile
#define _SI_SC_MagFields_Cyl3D_HeaderFile

#include <FieldsBase.hxx>

#include <FaceMagFldsBase_Cyl3D.hxx>
#include <SweptFaceMagFldsBase_Cyl3D.hxx>


class SI_SC_MagFields_Cyl3D : public FieldsBase
{
public:
  SI_SC_MagFields_Cyl3D();
  SI_SC_MagFields_Cyl3D(const FieldsDefineCntr* theCntr, 
		       PhysDataDefineRule theRule);

  virtual ~SI_SC_MagFields_Cyl3D();


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
  virtual void AddMagAlongAixs();


public:
  FaceMagFldsBase_Cyl3D* m_FaceMagFlds_Cyl3D;
  SweptFaceMagFldsBase_Cyl3D* m_SweptFaceMagFlds_Cyl3D;
}; 

#endif
