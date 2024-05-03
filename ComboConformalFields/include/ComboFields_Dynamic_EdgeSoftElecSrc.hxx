#ifndef _ComboFields_Dynamic_EdgeSoftElecSrc_HeaderFile
#define _ComboFields_Dynamic_EdgeSoftElecSrc_HeaderFile

#include <ComboFields_Dynamic_EdgeElecSrc.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_EdgeSoftElecSrc : public ComboFields_Dynamic_EdgeElecSrc
{
public:
  ComboFields_Dynamic_EdgeSoftElecSrc();
  ComboFields_Dynamic_EdgeSoftElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule);
  
  virtual ~ComboFields_Dynamic_EdgeSoftElecSrc();
  

public:
  virtual void Setup();
  virtual void Advance();

  virtual void Advance_SI_J(const Standard_Real si_scale);

protected:
  Standard_Integer m_PhysDataIndex;
};

#endif
