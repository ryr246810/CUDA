#ifndef _ZRDefine_HeaderFile
#define _ZRDefine_HeaderFile


#include <TxVector2D.h>
#include <TxVector.h>

#include <TxSlab2D.h>
#include <TxSlab.h>

#include <Standard_TypeDefine.hxx>
#include <UnitsSystemDef.hxx>


class ZRDefine
{

public:
  ZRDefine();
  ZRDefine(Standard_Real theXYZOrg[3], Standard_Integer zdir, Standard_Integer rdir);

  ~ZRDefine();

  Standard_Integer GetZDir() const;
  Standard_Integer GetRDir() const;
  Standard_Integer GetWorkPlaneDir() const;

  Standard_Integer GetZRDir_AccordingTo_XYZDir(const Standard_Integer aDir) const;
  Standard_Integer GetXYZDir_AccordingTo_ZRDir(const Standard_Integer aDir) const;

  const TxVector<Standard_Real>& GetZUnitVec() const;
  const TxVector<Standard_Real>& GetRUnitVec() const;
  const TxVector<Standard_Real>& GetWorkPlaneUnitVec() const;

  const TxVector<Standard_Real>& GetXYZUnitVecAccordingRZDir(Standard_Integer dir) const;

  const TxVector<Standard_Real>& GetXYZOrg() const;

  void Convert_ZR_to_XYZ(const Standard_Real theZRPnt[2], TxVector<Standard_Real>& theXYZ) const;
  void Convert_ZR_to_XYZ(const TxVector2D<Standard_Real>& theZRPnt, TxVector<Standard_Real>& theXYZ) const;

  void Convert_XYZ_to_ZR(const TxVector<Standard_Real>& theXYZ, Standard_Real theZRPnt[2]) const;
  void Convert_XYZ_to_ZR(const TxVector<Standard_Real>& theXYZ, TxVector2D<Standard_Real>& theZRPnt) const;

  void Convert_XYZ_to_ZR(const TxSlab<Standard_Real>& theXYZSlab, TxSlab2D<Standard_Real>& theZRSlab) const;

public:
  void ScaleAccordingUnitSystem(UnitsSystemDef* theUnitsSystem);

private:
  TxVector<Standard_Real> m_WorkPlaneUnitVec;
  TxVector<Standard_Real> m_ZUnitVec;
  TxVector<Standard_Real> m_RUnitVec;

  Standard_Integer m_ZDir;
  Standard_Integer m_RDir;
  Standard_Integer m_WorkPlanDir;

  TxVector<Standard_Real> m_XYZOrg;
};

#endif
