#include <ComboFields_Dynamic_VertexSoftElecSrc.hxx>

void 
ComboFields_Dynamic_VertexSoftElecSrc::
Advance()
{
  DynObj::Advance();
}


void 
ComboFields_Dynamic_VertexSoftElecSrc::
Advance_SI_J(const Standard_Real si_scale)
{
  exit(-1);
  double t =  GetCurTime();

  if(t > m_StartTime && t < m_EndTime){
    Standard_Real funcValue = m_tfuncPtr->operator()(t);
    for(Standard_Size i=0; i<m_Datas.size(); i++){
      m_Datas[i]->SetSweptPhysData(m_PhysDataIndex, funcValue);
    }
  }else{
    for(Standard_Size i=0; i<m_Datas.size(); i++){
      m_Datas[i]->SetSweptPhysData(m_PhysDataIndex, 0.0);
    }
  }

  DynObj::AdvanceSI(si_scale);
}
