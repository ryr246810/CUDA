#ifndef _GridMaterialDataDefine_HeaderFile
#define _GridMaterialDataDefine_HeaderFile

#include <Standard_TypeDefine.hxx>

typedef struct{
  Standard_Real m_eps_z;
  Standard_Real m_eps_r;
  Standard_Real m_eps_Phi;
  Standard_Real m_mu_z;
  Standard_Real m_mu_r;
  Standard_Real m_mu_Phi;
  Standard_Real m_sigma_z;
  Standard_Real m_sigma_r;
  Standard_Real m_sigma_Phi;
} ISOEMMatData;

#endif
