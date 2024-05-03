#ifndef _ComboFields_Dynamic_VertexSoftElecSrc_HeaderFile
#define _ComboFields_Dynamic_VertexSoftElecSrc_HeaderFile

#include <ComboFields_Dynamic_VertexElecSrc.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_VertexSoftElecSrc : public ComboFields_Dynamic_VertexElecSrc
{
public:
  ComboFields_Dynamic_VertexSoftElecSrc();
  ComboFields_Dynamic_VertexSoftElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule);
  
  virtual ~ComboFields_Dynamic_VertexSoftElecSrc();
  

public:
  virtual void Setup();
  virtual void Advance();

  virtual void Advance_SI_J(const Standard_Real si_scale);

protected:
  Standard_Integer m_PhysDataIndex;
};

#endif
