#ifndef _ComboFields_Dynamic_SrcBase_HeaderFile
#define _ComboFields_Dynamic_SrcBase_HeaderFile


#include <FieldsSrcBase.hxx>

// txbase includes
#include <TxHierAttribSet.h>


class ComboFields_Dynamic_SrcBase : public FieldsSrcBase
{
public:
  ComboFields_Dynamic_SrcBase();

  ComboFields_Dynamic_SrcBase(const FieldsDefineCntr* theCntr, 
			      PhysDataDefineRule theRule);

  virtual void SetAttrib(const TxHierAttribSet& ){}
  virtual ~ComboFields_Dynamic_SrcBase();

  void SetWorkDir(const std::string theDir);

  void SetPhiIndex(Standard_Integer phi_index) { m_PhiIndex = phi_index;}
  Standard_Integer GetPhiIndex() {return m_PhiIndex;}

public:
  virtual void Setup();
  virtual void Advance();


  virtual void Advance_SI_J(const Standard_Real si_scale);
  virtual void Advance_SI_MJ(const Standard_Real si_scale);

  virtual void Advance_SI_Elec_0(const Standard_Real si_scale);
  virtual void Advance_SI_Mag_0(const Standard_Real si_scale);

  virtual void Advance_SI_Elec_1(const Standard_Real si_scale);
  virtual void Advance_SI_Mag_1(const Standard_Real si_scale);

  virtual void Advance_SI_Elec_Damping_0(const Standard_Real si_scale, Standard_Real damping_scale);
  virtual void Advance_SI_Mag_Damping_0(const Standard_Real si_scale);

  virtual void Advance_SI_Elec_Damping_1(const Standard_Real si_scale, Standard_Real damping_scale);
  virtual void Get_Parameters(Standard_Real& Ebar, Standard_Real& Ebar2);
  virtual void Get_VBar(Standard_Real& VBar);
  virtual void Get_amp(Standard_Real** amp, Standard_Integer& amp_size);
  virtual void Get_Ptr(vector<GridEdgeData*>* MurEdgeDatas, vector<GridEdgeData*>* FreeEdgeDatas,
					             vector<GridVertexData*>* MurSweptEdgeDatas, vector<GridVertexData*>* FreeSweptEdgeDatas);
	
  virtual void Advance_SI_Mag_Damping_1(const Standard_Real si_scale);
  virtual void advance(Standard_Real scale);


protected:
  std::string m_WorkDir;
  Standard_Integer m_PhiIndex;
};

#endif
