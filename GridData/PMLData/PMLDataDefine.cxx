#include <math.h>
#include <PMLDataDefine.hxx>

#include <iostream>

PMLDataDefine::PMLDataDefine()
{
  m_PowOrder = 4;
  m_GeomProg = 2.15;

  m_SigmaRatio = 1.75;
  m_Alpha = 0.25;
  m_KappaMax = 4;
}


PMLDataDefine::~PMLDataDefine()
{
}


void PMLDataDefine::SetSigmaRatio(const Standard_Real _sigmaRatio)
{
  m_SigmaRatio = _sigmaRatio;
}

void PMLDataDefine::SetPowOrder(const Standard_Integer _mm)
{
  m_PowOrder = _mm;
}

void PMLDataDefine::SetGeomProg(const Standard_Real _data)
{
  m_GeomProg = _data;
}


void PMLDataDefine::SetAlpha(const Standard_Real _alpha)
{
  m_Alpha = _alpha;
}

void PMLDataDefine::SetKappaMax(const Standard_Real _kappaMax)
{
  m_KappaMax = _kappaMax;
}


void PMLDataDefine::SetMethodKey(Standard_Integer _data)
{
  m_MethodKey = _data;
}


Standard_Integer PMLDataDefine::GetMethodKey() const
{
  return m_MethodKey;
}


Standard_Real PMLDataDefine::GetSigmaRatio() const
{
  return m_SigmaRatio;
}

Standard_Real PMLDataDefine::GetPowOrder() const
{
  return m_PowOrder;
}


Standard_Real PMLDataDefine::GetGeomProg() const
{
  return m_GeomProg;
}


Standard_Real PMLDataDefine::GetAlpha() const
{
  return m_Alpha;
}

Standard_Real PMLDataDefine::GetKappaMax() const
{
  return m_KappaMax;
}


