#ifndef _ComboFields_Dynamic_VertexHardElecSrc_HeaderFile
#define _ComboFields_Dynamic_VertexHardElecSrc_HeaderFile

#include <ComboFields_Dynamic_VertexElecSrc.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_VertexHardElecSrc : public ComboFields_Dynamic_VertexElecSrc
{
public:
  ComboFields_Dynamic_VertexHardElecSrc();
  ComboFields_Dynamic_VertexHardElecSrc(const FieldsDefineCntr* theCntr, 
				      PhysDataDefineRule theRule);
  
  virtual ~ComboFields_Dynamic_VertexHardElecSrc();
  

public:
  virtual void Setup();
  virtual void Advance();

  virtual void Advance_SI_Elec_1(const Standard_Real si_scale);
  virtual void Advance_SI_Elec_Damping_1(const Standard_Real si_scale, const Standard_Real damping);
  virtual void Advance_SI_J(const Standard_Real si_scale);

protected:
  Standard_Integer m_PhysDataIndex,m_PhysJDataIndex;
};

#endif
