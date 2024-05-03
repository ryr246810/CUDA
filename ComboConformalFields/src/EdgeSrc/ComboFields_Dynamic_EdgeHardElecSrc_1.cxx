#include <ComboFields_Dynamic_EdgeHardElecSrc.hxx>

void 
ComboFields_Dynamic_EdgeHardElecSrc::
Advance()
{
  DynObj::Advance();
}


void 
ComboFields_Dynamic_EdgeHardElecSrc::
Advance_SI_Elec_1(const Standard_Real si_scale)
{
  double t =  GetCurTime();
  DynObj::AdvanceSI(si_scale);

  if(m_Datas.empty()) return;
  Standard_Size nb = m_Datas.size();
  if(t > m_StartTime && t < m_EndTime){
    Standard_Real funcValue = m_tfuncPtr->operator()(t);
    for(Standard_Size i=0; i<nb; i++){
      m_Datas[i]->SetPhysData(m_PhysDataIndex, funcValue);
    }
  }
}
