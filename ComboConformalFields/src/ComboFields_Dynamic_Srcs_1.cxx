#include <ComboFields_Dynamic_Srcs.hxx>

void 
ComboFields_Dynamic_Srcs::
Advance()
{
  DynObj::Advance();
};


void 
ComboFields_Dynamic_Srcs::
Advance_SI_J(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_J(si_scale);
  }
}


void 
ComboFields_Dynamic_Srcs::
Advance_SI_MJ(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_MJ(si_scale);
  }
}



void 
ComboFields_Dynamic_Srcs::
Advance_SI_Elec_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Elec_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Elec_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Elec_1(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Mag_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Mag_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Mag_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Mag_1(si_scale);
  }
}




void 
ComboFields_Dynamic_Srcs::
Advance_SI_Elec_Damping_0(const Standard_Real si_scale, Standard_Real damping_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Elec_Damping_0(si_scale, damping_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Elec_Damping_1(const Standard_Real si_scale, Standard_Real damping_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Elec_Damping_1(si_scale, damping_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Mag_Damping_0(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Mag_Damping_0(si_scale);
  }
}

void 
ComboFields_Dynamic_Srcs::
Advance_SI_Mag_Damping_1(const Standard_Real si_scale)
{
  for(Standard_Size i=0; i<m_Srcs.size(); i++){
    m_Srcs[i]->Advance_SI_Mag_Damping_1(si_scale);
  }
}
