#ifndef _ComboFields_Dynamic_EdgeHardElecSrc_HeaderFile
#define _ComboFields_Dynamic_EdgeHardElecSrc_HeaderFile

#include <ComboFields_Dynamic_EdgeElecSrc.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_EdgeHardElecSrc : public ComboFields_Dynamic_EdgeElecSrc
{
public:
  ComboFields_Dynamic_EdgeHardElecSrc();
  ComboFields_Dynamic_EdgeHardElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule);
  
  virtual ~ComboFields_Dynamic_EdgeHardElecSrc();
  

public:
  virtual void Setup();
  virtual void Advance();

  virtual void Advance_SI_Elec_1(const Standard_Real si_scale);

protected:
  Standard_Integer m_PhysDataIndex;
};

#endif
