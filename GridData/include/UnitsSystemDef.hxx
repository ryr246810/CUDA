#ifndef _UnitsSystemDef_Headerfile
#define _UnitsSystemDef_Headerfile

#include <Standard_TypeDefine.hxx>
#include <math.h>

class UnitsSystemDef
{

public:
  UnitsSystemDef()
  {
    m_UnitScaleOfLength = 0;
    m_UnitScaleOfTime = 0;
    m_UnitScaleOfMass = 0;

    m_RealUnitScaleOfLength = 0;
    m_RealUnitScaleOfTime = 0;
    m_RealUnitScaleOfMass = 0;
  };

  ~UnitsSystemDef()
  {
  };


public:
  void SetUnitScaleOfLength(const Standard_Integer _data)
  {
    m_UnitScaleOfLength = _data;
    m_RealUnitScaleOfLength = pow(10.0, m_UnitScaleOfLength);
  };
  void SetUnitScaleOfTime(const Standard_Integer _data)
  {
    m_UnitScaleOfTime = _data;
    m_RealUnitScaleOfTime = pow(10.0, m_UnitScaleOfTime);
  };
  void SetUnitScaleOfMass(const Standard_Integer _data)
  {
    m_UnitScaleOfMass = _data;
    m_RealUnitScaleOfMass = pow(10.0, m_UnitScaleOfMass);
  };


public:
  Standard_Integer GetUnitScaleOfLength() const
  {
    return m_UnitScaleOfLength;
  };

  Standard_Integer GetUnitScaleOfTime() const
  {
    return m_UnitScaleOfTime;
  };

  Standard_Integer GetUnitScaleOfMass() const
  {
    return m_UnitScaleOfMass;
  };


public:
  Standard_Real GetRealUnitScaleOfLength() const
  {
    return m_RealUnitScaleOfLength;
  };

  Standard_Real GetRealUnitScaleOfTime() const
  {
    return m_RealUnitScaleOfTime;
  };

  Standard_Real GetRealUnitScaleOfMass() const
  {
    return m_RealUnitScaleOfMass;
  };

private:
  Standard_Integer m_UnitScaleOfLength;
  Standard_Integer m_UnitScaleOfTime;
  Standard_Integer m_UnitScaleOfMass;


  Standard_Real m_RealUnitScaleOfLength;
  Standard_Real m_RealUnitScaleOfTime;
  Standard_Real m_RealUnitScaleOfMass;
};


#endif
