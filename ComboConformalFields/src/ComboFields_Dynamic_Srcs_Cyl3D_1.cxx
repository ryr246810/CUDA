#include <ComboFields_Dynamic_Srcs_Cyl3D.hxx>

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance()
{
  DynObj::Advance();
};


void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_J(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_J(si_scale);
  }
}


void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_MJ(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_MJ(si_scale);
  }
}



void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Elec_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Elec_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Elec_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Elec_1(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Mag_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Mag_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Mag_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Mag_1(si_scale);
  }
}




void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Elec_Damping_0(const Standard_Real si_scale, Standard_Real damping_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Elec_Damping_0(si_scale, damping_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Get_SrcData(Standard_Real** amp, Standard_Integer& amp_size)
{
  Standard_Real* ampTmp;
  m_Srcs_Cyl3D[0]->Get_amp(&ampTmp, amp_size);
  *amp = ampTmp;
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Get_SrcDataVec(int idx, vector<GridEdgeData*>* MurEdgeDatas, vector<GridEdgeData*>* FreeEdgeDatas,
               vector<GridVertexData*>* MurSweptEdgeDatas, vector<GridVertexData*>* FreeSweptEdgeDatas)
{
  vector<GridEdgeData*> murEdgeDatas;
	vector<GridEdgeData*> freeEdgeDatas;
	vector<GridVertexData*> murSweptEdgeDatas;
	vector<GridVertexData*> freeSweptEdgeDatas;

  m_Srcs_Cyl3D[idx]->Get_Ptr(&murEdgeDatas, &freeEdgeDatas, &murSweptEdgeDatas, &freeSweptEdgeDatas);

  *MurEdgeDatas = murEdgeDatas;
	*FreeEdgeDatas = freeEdgeDatas;
	*MurSweptEdgeDatas = murSweptEdgeDatas;
	*FreeSweptEdgeDatas = freeSweptEdgeDatas;
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
addCurrTime(int idx, Standard_Real scale)
{
  m_Srcs_Cyl3D[idx]->advance(scale);
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Get_Parameters(int idx, Standard_Real& Ebar, Standard_Real& Ebar2)
{

  m_Srcs_Cyl3D[idx]->Get_Parameters(Ebar, Ebar2);
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Get_VBar(int idx, Standard_Real& VBar)
{

  m_Srcs_Cyl3D[idx]->Get_VBar(VBar);
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Elec_Damping_1(const Standard_Real si_scale, Standard_Real damping_scale)
{
  for(Standard_Size i = 0; i < m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Elec_Damping_1(si_scale, damping_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Mag_Damping_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Mag_Damping_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs_Cyl3D::
Advance_SI_Mag_Damping_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs_Cyl3D.size(); i++){
    m_Srcs_Cyl3D[i]->Advance_SI_Mag_Damping_1(si_scale);
  }
}
