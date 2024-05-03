#ifndef _ComboFields_Dynamic_SweptFaceSoftMagSrc_HeaderFile
#define _ComboFields_Dynamic_SweptFaceSoftMagSrc_HeaderFile

#include <ComboFields_Dynamic_SweptFaceMagSrc.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_SweptFaceSoftMagSrc : public ComboFields_Dynamic_SweptFaceMagSrc
{
public:
  ComboFields_Dynamic_SweptFaceSoftMagSrc();
  ComboFields_Dynamic_SweptFaceSoftMagSrc(const FieldsDefineCntr* theCntr, 
					  PhysDataDefineRule theRule);
  
  virtual ~ComboFields_Dynamic_SweptFaceSoftMagSrc();
  

public:
  virtual void Setup();
  virtual void Advance();

  virtual void Advance_SI_MJ(const Standard_Real si_scale);

protected:
  Standard_Integer m_PhysDataIndex;
};

#endif
