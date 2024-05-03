#ifndef _DataBase_Headerfile
#define _DataBase_Headerfile

#include <GeomDataBase.hxx>

class DataBase:public GeomDataBase
{
public:
  DataBase();
  DataBase(Standard_Integer m_Mark);
  ~DataBase();


  virtual void Setup();
  virtual void SetupMaterialData();

  /**************************************************************/
public:
  void AddMaterialType(const Standard_Integer _materialtype);
  void DelMaterialType(const Standard_Integer _materialtype);
  bool IsMaterialType(const Standard_Integer _materialtype) const;

  void SetMaterialType(const Standard_Integer _materialtype);
  Standard_Integer GetMaterialType() const;
  void ResetMaterialType();
  void ResetEMMaterialType();
  void ResetPtclMaterialType();

  Standard_Integer GetEMMaterialType() const;
  Standard_Integer GetPtclMaterialType() const;
  /**************************************************************/


  /**************************************************************/
public:
  void ClearMaterialData();
  bool IsMaterialDataDefined() const;

  void SetEpsilon(const Standard_Real _epsilon);
  void SetMu(const Standard_Real _sigma);
  void SetSigma(const Standard_Real _sigma);

  Standard_Real GetEpsilon() const;
  Standard_Real GetSigma() const;
  Standard_Real GetMu() const;
  Standard_Real GetEpsilonInv() const;
  Standard_Real GetSigmaInv() const;
  Standard_Real GetMuInv() const;
  /**************************************************************/


  /**************************************************************/
public:
  void SetupPhysData(Standard_Integer _datanum);
  void ClearPhysData();
  void ZeroPhysDatas();

  bool IsPhysDataDefined()  const;

  void SetPhysData(Standard_Integer _index, Standard_Real _value);
  void AddPhysData(Standard_Integer _index, Standard_Real _value);
  void SubtractPhysData(Standard_Integer _index, Standard_Real _value);

  Standard_Size    GetPhysDataNum() const;
  Standard_Real*   GetPhysData();
  Standard_Real    GetPhysData(Standard_Integer _index) const;
  Standard_Real *  GetPhysDataPtr(Standard_Integer _index);
  /**************************************************************/


  /**************************************************************/
public:
  void SetupSweptPhysData(Standard_Integer _datanum);
  void ClearSweptPhysData();
  void ZeroSweptPhysDatas();
 
  bool IsSweptPhysDataDefined()  const;

  void SetSweptPhysData(Standard_Integer _index, Standard_Real _value);
  void AddSweptPhysData(Standard_Integer _index, Standard_Real _value);
  void SubtractSweptPhysData(Standard_Integer _index, Standard_Real _value);

  Standard_Size    GetSweptPhysDataNum() const;
  Standard_Real*   GetSweptPhysData();
  Standard_Real    GetSweptPhysData(Standard_Integer _index) const;
  Standard_Real *  GetSweptPhysDataPtr(Standard_Integer _index);
  /**************************************************************/


public:
  bool BeInPMLRegion() const;
  bool IsPMLDataDefined() const;

  void SetupPMLData();
  void LocateMemeoryForPMLData();

  void ResetPMLData();
  void ClearPMLData();

  void SetPMLSigma(Standard_Integer _dir, Standard_Real _pmlsigma);
  Standard_Real GetPMLSigma(Standard_Integer _dir);

  void SetPMLAlpha(Standard_Integer _dir, Standard_Real _pmlalpha);
  Standard_Real GetPMLAlpha(Standard_Integer _dir);

  void SetPMLKappa(Standard_Integer _dir, Standard_Real _pmlkappa);
  Standard_Real GetPMLKappa(Standard_Integer _dir);

  void SetPML_a(Standard_Integer _dir, Standard_Real _pmla);
  Standard_Real GetPML_a(Standard_Integer _dir);

  void SetPML_b(Standard_Integer _dir, Standard_Real _pmlb);
  Standard_Real GetPML_b(Standard_Integer _dir);
  
  void ResetPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr);
  void ResetPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr, Standard_Integer _isElecEdge);
  void ResetSweptPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr);
  void ResetSweptPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr, Standard_Integer _isElecEdge);
  void CleanPhysDataPtr();
  void CleanSweptPhysDataPtr();


protected:
  Standard_Integer m_MaterialType;

  Standard_Integer  m_PhysDataNum;  // for freespace region E/B, J/M, AE, BE, PRE; for pml region E/B, J/M, AE, BE, PRE, PE1, PE2
  Standard_Real* m_PhysData;

  Standard_Integer  m_SweptPhysDataNum;  // for freespace region E/B, J/M, AE, BE, PRE;   for pml region E/B, J/M, AE, BE, PRE, PE1, PE2
  Standard_Real* m_SweptPhysData;

protected:
  Standard_Real* m_MaterialData;  // epsilon mu sigma
  Standard_Real* m_MaterialDataInv; // epsilon mu sigma inv
  Standard_Real* m_PMLData;       // sigma(z,r,phi), alpha(z,r,phi), kappa(z,r,phi), a(z,r,phi), b(z,r,phi)

};

#endif
