#include <SI_SC_ComboEMFields.hxx>
#include <stdlib.h>

#include <ComboFields_Dynamic_Srcs.hxx>

SI_SC_ComboEMFields::
    SI_SC_ComboEMFields()
    : Dynamic_ComboEMFieldsBase()
{
  m_Order = 1;
  m_Step = 1;

  m_b = NULL;
  m_bb = NULL;

  m_Damping = 0.0;

  m_ECntrFields = NULL;
  m_MCntrFields = NULL;

  m_ECPMLFields = NULL;
  m_MCPMLFields = NULL;
  m_EMurFields = NULL;
}

SI_SC_ComboEMFields::
    SI_SC_ComboEMFields(const FieldsDefineCntr *_cntr)
    : Dynamic_ComboEMFieldsBase(_cntr)
{
}

SI_SC_ComboEMFields::
    ~SI_SC_ComboEMFields()
{
  if (m_ECntrFields != NULL)
    delete m_ECntrFields;
  if (m_MCntrFields != NULL)
    delete m_MCntrFields;

  if (m_ECPMLFields != NULL)
    delete m_ECPMLFields;
  if (m_MCPMLFields != NULL)
    delete m_MCPMLFields;
  if (m_EMurFields != NULL)
    delete m_EMurFields;

  if (m_b != NULL)
    delete[] m_b;
  if (m_bb != NULL)
    delete[] m_bb;
}

void SI_SC_ComboEMFields::
    SetOrder(const Standard_Integer theOrder)
{
  m_Order = theOrder;

  if (theOrder == 1)
  {
    m_Step = 1;
    m_b = new Standard_Real[m_Step];
    m_bb = new Standard_Real[m_Step];

    m_b[0] = 1.0;
    m_bb[0] = 1.0;
  }
  else if (theOrder == 2)
  {

    m_Step = 1;
    m_b = new Standard_Real[m_Step];
    m_bb = new Standard_Real[m_Step];

    m_b[0] = 1.0;
    m_bb[0] = 1.0;
    /*
    m_Step = 2;
    m_b = new Standard_Real[m_Step];
    m_bb = new Standard_Real[m_Step];

    m_b[0] = 0.0;
    m_b[1] = 1.0;

    m_bb[0] = 0.5;
    m_bb[1] = 0.5;
    //*/
  }
  else if (theOrder == 3)
  {
    m_Step = 3;
    m_b = new Standard_Real[m_Step];
    m_bb = new Standard_Real[m_Step];

    /*
    m_bb[0] = 0.9196615230173999;
    m_bb[1] = 0.25/m_bb[0]-0.5*m_bb[0];
    m_bb[2] = 1.0-m_bb[0]-m_bb[1];

    m_b[0] = m_bb[2];
    m_b[1] = m_bb[1];
    m_b[2] = m_bb[0];
    //*/

    m_bb[0] = 2.0 / 3.0;
    m_bb[1] = -2.0 / 3.0;
    m_bb[2] = 1.0;

    m_b[0] = 7.0 / 24.0;
    m_b[1] = 3.0 / 4.0;
    m_b[2] = -1.0 / 24.0;
  }
  else if (theOrder == 4)
  {
    m_Step = 6;
    m_b = new Standard_Real[m_Step];
    m_bb = new Standard_Real[m_Step];

    m_bb[0] = 0.2167979108466032;
    m_bb[1] = -0.0283101143283301;
    m_bb[2] = 0.3901418904713324;
    m_bb[3] = -0.2414087476423302;
    m_bb[4] = 0.5908564573813148;
    m_bb[5] = 0.0719226032714098;

    m_b[0] = m_bb[5];
    m_b[1] = m_bb[4];
    m_b[2] = m_bb[3];
    m_b[3] = m_bb[2];
    m_b[4] = m_bb[1];
    m_b[5] = m_bb[0];
  }
  else
  {
    cout << "error ------------SI_SC_ComboEMFields::SetOrder-------------should be not larger than 4" << endl;
    exit(1);
  }
}

void SI_SC_ComboEMFields::
    SetDamping(const Standard_Real theDamping)
{
  m_Damping = theDamping;
}

void SI_SC_ComboEMFields::
    Setup()
{
  SetupCntrElecFields();
  SetupCntrMagFields();

  SetupCPMLElecFields();
  SetupCPMLMagFields();
  SetupMurElecFields();

  Dynamic_ComboEMFieldsBase::Setup();
}

void SI_SC_ComboEMFields::
    Advance()
{
  m_FldSrcs->Advance();

  for (Standard_Integer i = 0; i < m_Step; i++)
  {
    m_FldSrcs->Advance_SI_MJ(m_b[i]);
    m_FldSrcs->Advance_SI_Mag_0(m_b[i]);
    m_MCntrFields->Advance_SI(m_b[i]);
    m_MCPMLFields->Advance_SI(m_b[i]);
    m_FldSrcs->Advance_SI_Mag_1(m_b[i]);

    m_FldSrcs->Advance_SI_J(m_b[i]);
    m_FldSrcs->Advance_SI_Elec_0(m_b[i]);
    m_ECntrFields->Advance_SI(m_bb[i]);
    m_ECPMLFields->Advance_SI(m_bb[i]);
    m_EMurFields->Advance_SI(m_bb[i]);
    m_FldSrcs->Advance_SI_Elec_1(m_b[i]);
  }

  m_MCntrFields->Advance();
  m_MCPMLFields->Advance();
  m_ECntrFields->Advance();
  m_ECPMLFields->Advance();

  DynObj::Advance();
}

void SI_SC_ComboEMFields::
    BuildCUDADatas()
{
}

void SI_SC_ComboEMFields::
    TransDgnData()
{
}

void SI_SC_ComboEMFields::
    InitMatrixData()
{
}

void SI_SC_ComboEMFields::
    CleanCUDADatas()
{
}

void SI_SC_ComboEMFields::
    AdvanceWithDamping()
{
  m_FldSrcs->Advance();

  for (Standard_Integer i = 0; i < m_Step; i++)
  {
    m_FldSrcs->Advance_SI_MJ(m_b[i]);
    m_FldSrcs->Advance_SI_Mag_Damping_0(m_b[i]);
    m_MCntrFields->Advance_SI_Damping(m_b[i]);
    m_MCPMLFields->Advance_SI_Damping(m_b[i]);
    m_FldSrcs->Advance_SI_Mag_Damping_1(m_b[i]);

    m_FldSrcs->Advance_SI_J(m_b[i]);
    m_FldSrcs->Advance_SI_Elec_Damping_0(m_b[i], m_Damping);
    m_ECntrFields->Advance_SI_Damping(m_bb[i], m_Damping);
    m_ECPMLFields->Advance_SI_Damping(m_bb[i], m_Damping);
    m_EMurFields->Advance_SI_Damping(m_bb[i], m_Damping);
    m_FldSrcs->Advance_SI_Elec_Damping_1(m_b[i], m_Damping);
  }

  m_MCntrFields->Advance();
  m_MCPMLFields->Advance();
  m_ECntrFields->Advance();
  m_ECPMLFields->Advance();

  DynObj::Advance();
}

bool SI_SC_ComboEMFields::
    IsPhysDataMemoryLocated()
{
  cout << "-----------------------IsPhysDataMemoryLocated------------------------1" << endl;
  bool result = false;

  if (m_ECntrFields->IsPhysDataMemoryLocated() &&
      m_MCntrFields->IsPhysDataMemoryLocated() &&
      m_ECPMLFields->IsPhysDataMemoryLocated() &&
      m_MCPMLFields->IsPhysDataMemoryLocated())
  {
    result = true;
  }
  // cout<<"-----------------------IsPhysDataMemoryLocated------------------------2"<<endl;
  if (result)
    cout << "IsPhysDataMemoryLocated is ok" << endl;
  else
    cout << "IsPhysDataMemoryLocated is not ok" << endl;
  return result;
}

void SI_SC_ComboEMFields::
    ZeroPhysDatas()
{
  cout << "----------------------ZeroPhysDatas----------------------1" << endl;
  if (IsPhysDataMemoryLocated())
  {
    cout << "----------------------ZeroPhysDatas----------------------2" << endl;
    m_ECntrFields->ZeroPhysDatas();
    cout << "----------------------ZeroPhysDatas----------------------3" << endl;
    m_MCntrFields->ZeroPhysDatas();
    cout << "----------------------ZeroPhysDatas----------------------4" << endl;
    m_ECPMLFields->ZeroPhysDatas();
    cout << "----------------------ZeroPhysDatas----------------------5" << endl;
    m_MCPMLFields->ZeroPhysDatas();
    cout << "----------------------ZeroPhysDatas----------------------6" << endl;
    m_EMurFields->ZeroPhysDatas();
  }
}

void SI_SC_ComboEMFields::
    SetupCntrElecFields()
{
  set<Standard_Integer> theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet(theBndsDefine);

  m_ECntrFields = new SI_SC_ElecFields(GetFldsDefCntr(), EXCLUDING);
  m_ECntrFields->SetMaterials(theBndsDefine);
  Standard_Real dt = this->GetDelTime();
  m_ECntrFields->SetDelTime(dt);

  m_ECntrFields->Setup();
}

void SI_SC_ComboEMFields::
    SetupCntrMagFields()
{
  set<Standard_Integer> theBndsDefine;
  GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagMaterialSet(theBndsDefine);

  m_MCntrFields = new SI_SC_MagFields(GetFldsDefCntr(), EXCLUDING);
  m_MCntrFields->SetMaterials(theBndsDefine);

  Standard_Real dt = this->GetDelTime();
  m_MCntrFields->SetDelTime(dt);

  m_MCntrFields->Setup();
}

void SI_SC_ComboEMFields::
    SetupCPMLElecFields()
{
  m_ECPMLFields = new SI_SC_CPML_ElecFields(GetFldsDefCntr());
  Standard_Real dt = this->GetDelTime();
  m_ECPMLFields->SetDelTime(dt);

  m_ECPMLFields->Setup();
  m_ECPMLFields->Setup_PML_a_b();
}

void SI_SC_ComboEMFields::
    SetupCPMLMagFields()
{
  m_MCPMLFields = new SI_SC_CPML_MagFields(GetFldsDefCntr());
  Standard_Real dt = this->GetDelTime();
  m_MCPMLFields->SetDelTime(dt);

  m_MCPMLFields->Setup();
  m_MCPMLFields->Setup_PML_a_b();
}

void SI_SC_ComboEMFields::
    SetupMurElecFields()
{
  m_EMurFields = new SI_SC_Mur_ElecFieldsSet(GetFldsDefCntr());
  Standard_Real dt = this->GetDelTime();
  m_EMurFields->SetDelTime(dt);

  m_EMurFields->Setup();
}

void SI_SC_ComboEMFields::
    Write_PML_Inf()
{
  ostringstream sstr1;
  sstr1 << "CPML_E";
  sstr1 << ".txt";
  string oFileName1 = sstr1.str();
  ofstream txtStream1(oFileName1.c_str());

  m_ECPMLFields->Write_PML_a_b(txtStream1);

  ostringstream sstr2;
  sstr2 << "CPML_M";
  sstr2 << ".txt";
  string oFileName2 = sstr2.str();
  ofstream txtStream2(oFileName2.c_str());

  m_MCPMLFields->Write_PML_a_b(txtStream2);
}

vector<GridEdgeData *> &
SI_SC_ComboEMFields::
    GetCntrElecEdges()
{
  return m_ECntrFields->GetEdgeDatas();
}

vector<GridVertexData *> &
SI_SC_ComboEMFields::
    GetCntrElecVertices()
{
  return m_ECntrFields->GetVertexDatas();
}
