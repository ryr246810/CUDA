#ifndef _ComboFields_Dynamic_Srcs_HeaderFile
#define _ComboFields_Dynamic_Srcs_HeaderFile

#include  <FieldsDefineBase.hxx>
#include  <ComboFields_Dynamic_SrcBase.hxx>


class ComboFields_Dynamic_Srcs: public FieldsDefineBase
{
public:
  ComboFields_Dynamic_Srcs();
  ComboFields_Dynamic_Srcs(const FieldsDefineCntr* theCntr);
  virtual ~ComboFields_Dynamic_Srcs();


public:
  void SetAttrib(const string& theWorkDir,
		 const TxHierAttribSet& theFaceBndTha);


  void Append(ComboFields_Dynamic_SrcBase* _oneNewSrc);

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
  virtual void Advance_SI_Mag_Damping_1(const Standard_Real si_scale);

  virtual void Setup();

protected:
  vector<ComboFields_Dynamic_SrcBase*> m_Srcs;
};


#endif
