#include <ComboFields_Dynamic_VertexHardElecSrc.hxx>

void 
ComboFields_Dynamic_VertexHardElecSrc::
Advance()
{
  DynObj::Advance();
}


void 
ComboFields_Dynamic_VertexHardElecSrc::
Advance_SI_Elec_Damping_1(const Standard_Real si_scale, const Standard_Real damping)
{
  double t =  GetCurTime();
  DynObj::AdvanceSI(si_scale);
  if(m_Datas.empty()) return;
  Standard_Size nb = m_Datas.size();
  if(t > m_StartTime && t < m_EndTime){
    Standard_Real funcValue = m_tfuncPtr->operator()(t);
    for(Standard_Size i=0; i<nb; i++){
      m_Datas[i]->SetSweptPhysData(m_PhysDataIndex, funcValue);
    }
  }
}

void 
ComboFields_Dynamic_VertexHardElecSrc::
Advance_SI_Elec_1(const Standard_Real si_scale)
{
  /*double t =  GetCurTime();
  DynObj::AdvanceSI(si_scale);

	cout<<"ooooooooooooo"<<endl;
	getchar();
  if(m_Datas.empty()) return;
  Standard_Size nb = m_Datas.size();
  if(t > m_StartTime && t < m_EndTime){
    Standard_Real funcValue = m_tfuncPtr->operator()(t);
    for(Standard_Size i=0; i<nb; i++){
      m_Datas[i]->SetSweptPhysData(m_PhysDataIndex, funcValue);
    }
  }*/
}

void 
ComboFields_Dynamic_VertexHardElecSrc::
Advance_SI_J(const Standard_Real si_scale)
{
  exit(-1);
  double t =  GetCurTime();
  DynObj::AdvanceSI(si_scale);

  if(m_Datas.empty()) return;
  Standard_Size nb = m_Datas.size();
  if(t > m_StartTime && t < m_EndTime){
    Standard_Real funcValue = m_tfuncPtr->operator()(t);
    for(Standard_Size i=0; i<nb; i++){
      m_Datas[i]->SetSweptPhysData(m_PhysDataIndex, funcValue);
    }
  }
}
