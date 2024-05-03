#include <SI_SC_ComboEMFields_Cyl3D.hxx>
#include <stdlib.h>

#include <ComboFields_Dynamic_Srcs_Cyl3D.hxx>

SI_SC_ComboEMFields_Cyl3D::
    SI_SC_ComboEMFields_Cyl3D()
    : Dynamic_ComboEMFieldsBase()
{
    m_Order = 1;
    m_Step = 1;

    m_b = NULL;
    m_bb = NULL;

    m_Damping = 0.0;

    m_ECntrFields_Cyl3D = NULL;
    m_MCntrFields_Cyl3D = NULL;

    m_ECPMLFields_Cyl3D = NULL;
    m_MCPMLFields_Cyl3D = NULL;
    m_EMurFields_Cyl3D = NULL;
}

SI_SC_ComboEMFields_Cyl3D::
    SI_SC_ComboEMFields_Cyl3D(const FieldsDefineCntr *_cntr)
    : Dynamic_ComboEMFieldsBase(_cntr)
{
}

SI_SC_ComboEMFields_Cyl3D::
    ~SI_SC_ComboEMFields_Cyl3D()
{
    if (m_ECntrFields_Cyl3D != NULL)
        delete m_ECntrFields_Cyl3D;
    if (m_MCntrFields_Cyl3D != NULL)
        delete m_MCntrFields_Cyl3D;

    if (m_ECPMLFields_Cyl3D != NULL)
        delete m_ECPMLFields_Cyl3D;
    if (m_MCPMLFields_Cyl3D != NULL)
        delete m_MCPMLFields_Cyl3D;
    if (m_EMurFields_Cyl3D != NULL)
        delete m_EMurFields_Cyl3D;

    if (m_b != NULL)
        delete[] m_b;
    if (m_bb != NULL)
        delete[] m_bb;
}

void SI_SC_ComboEMFields_Cyl3D::
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

void SI_SC_ComboEMFields_Cyl3D::
    SetDamping(const Standard_Real theDamping)
{
    m_Damping = theDamping;
}

void SI_SC_ComboEMFields_Cyl3D::
    Setup()
{
    SetupCntrElecFields();
    SetupCntrMagFields();

    SetupCPMLElecFields();
    SetupCPMLMagFields();
    SetupMurElecFields();

    Dynamic_ComboEMFieldsBase::Setup();
}

void SI_SC_ComboEMFields_Cyl3D::
    Advance()
{
    m_FldSrcs_Cyl3D->Advance();

    for (Standard_Integer i = 0; i < m_Step; i++)
    {
        m_FldSrcs_Cyl3D->Advance_SI_MJ(m_b[i]);
        m_FldSrcs_Cyl3D->Advance_SI_Mag_0(m_b[i]);
        m_MCntrFields_Cyl3D->Advance_SI(m_b[i]);
        m_MCPMLFields_Cyl3D->Advance_SI(m_b[i]);
        m_FldSrcs_Cyl3D->Advance_SI_Mag_1(m_b[i]);

        m_FldSrcs_Cyl3D->Advance_SI_J(m_b[i]);
        m_FldSrcs_Cyl3D->Advance_SI_Elec_0(m_b[i]);
        m_ECntrFields_Cyl3D->Advance_SI(m_bb[i]);
        m_ECPMLFields_Cyl3D->Advance_SI(m_bb[i]);
        m_EMurFields_Cyl3D->Advance_SI(m_bb[i]);
        m_FldSrcs_Cyl3D->Advance_SI_Elec_1(m_b[i]);
    }

    m_MCntrFields_Cyl3D->Advance();
    m_MCPMLFields_Cyl3D->Advance();
    m_ECntrFields_Cyl3D->Advance();
    m_ECPMLFields_Cyl3D->Advance();

    DynObj::Advance();
}

void SI_SC_ComboEMFields_Cyl3D::
    BuildCUDADatas()
{
}

void SI_SC_ComboEMFields_Cyl3D::
    TransDgnData()
{
}

void SI_SC_ComboEMFields_Cyl3D::
    InitMatrixData()
{
}

void SI_SC_ComboEMFields_Cyl3D::
    CleanCUDADatas()
{
}

void SI_SC_ComboEMFields_Cyl3D::
    AdvanceWithDamping()
{
    m_FldSrcs_Cyl3D->Advance(); // 无差异

    for (Standard_Integer i = 0; i < m_Step; i++)
    {
        m_FldSrcs_Cyl3D->Advance_SI_MJ(m_b[i]);            // 无差异
        m_FldSrcs_Cyl3D->Advance_SI_Mag_Damping_0(m_b[i]); // 无内容

        m_MCntrFields_Cyl3D->Advance_SI_Damping(m_b[i]); // 此处考虑Near
        m_MCPMLFields_Cyl3D->Advance_SI_Damping(m_b[i]); // 此处考虑Near

        m_FldSrcs_Cyl3D->Advance_SI_Mag_Damping_1(m_b[i]); // 无内容

        m_FldSrcs_Cyl3D->Advance_SI_J(m_b[i]);                         // 无差异
        m_FldSrcs_Cyl3D->Advance_SI_Elec_Damping_0(m_b[i], m_Damping); // 无内容

        m_ECntrFields_Cyl3D->Advance_SI_Damping(m_bb[i], m_Damping); // 此处非常多不一样的
        m_ECPMLFields_Cyl3D->Advance_SI_Damping(m_bb[i], m_Damping); // 此处非常多不一样的
        m_EMurFields_Cyl3D->Advance_SI_Damping(m_bb[i], m_Damping);  // 无差异

        m_FldSrcs_Cyl3D->Advance_SI_Elec_Damping_1(m_b[i], m_Damping); // 无差异
    }

    m_MCntrFields_Cyl3D->Advance(); // 无差异
    m_MCPMLFields_Cyl3D->Advance(); // 无差异
    m_ECntrFields_Cyl3D->Advance(); // 无差异
    m_ECPMLFields_Cyl3D->Advance(); // 无差异
    m_EMurFields_Cyl3D->Advance();  // 无差异

    DynObj::Advance(); // 无差异
}

bool SI_SC_ComboEMFields_Cyl3D::
    IsPhysDataMemoryLocated()
{
    bool result = false;

    if (m_ECntrFields_Cyl3D->IsPhysDataMemoryLocated() &&
        m_MCntrFields_Cyl3D->IsPhysDataMemoryLocated() &&
        m_ECPMLFields_Cyl3D->IsPhysDataMemoryLocated() &&
        m_MCPMLFields_Cyl3D->IsPhysDataMemoryLocated())
    {
        result = true;
    }

    return result;
}

void SI_SC_ComboEMFields_Cyl3D::
    ZeroPhysDatas()
{
    if (IsPhysDataMemoryLocated())
    {
        m_ECntrFields_Cyl3D->ZeroPhysDatas();
        m_MCntrFields_Cyl3D->ZeroPhysDatas();
        m_ECPMLFields_Cyl3D->ZeroPhysDatas();
        m_MCPMLFields_Cyl3D->ZeroPhysDatas();

        m_EMurFields_Cyl3D->ZeroPhysDatas();
    }
}

void SI_SC_ComboEMFields_Cyl3D::
    SetupCntrElecFields()
{
    set<Standard_Integer> theBndsDefine;
    GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet(theBndsDefine);

    m_ECntrFields_Cyl3D = new SI_SC_ElecFields_Cyl3D(GetFldsDefCntr(), EXCLUDING);
    m_ECntrFields_Cyl3D->SetMaterials(theBndsDefine);
    Standard_Real dt = this->GetDelTime();
    m_ECntrFields_Cyl3D->SetDelTime(dt);

    m_ECntrFields_Cyl3D->Setup();
}

void SI_SC_ComboEMFields_Cyl3D::
    SetupCntrMagFields()
{
    set<Standard_Integer> theBndsDefine;
    GetFldsDefCntr()->GetFieldsDefineRules()->GetBndMagMaterialSet(theBndsDefine);

    m_MCntrFields_Cyl3D = new SI_SC_MagFields_Cyl3D(GetFldsDefCntr(), EXCLUDING);
    m_MCntrFields_Cyl3D->SetMaterials(theBndsDefine);

    Standard_Real dt = this->GetDelTime();
    m_MCntrFields_Cyl3D->SetDelTime(dt);

    m_MCntrFields_Cyl3D->Setup();
}

void SI_SC_ComboEMFields_Cyl3D::
    SetupCPMLElecFields()
{
    m_ECPMLFields_Cyl3D = new SI_SC_CPML_ElecFields_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_ECPMLFields_Cyl3D->SetDelTime(dt);

    m_ECPMLFields_Cyl3D->Setup();
    m_ECPMLFields_Cyl3D->Setup_PML_a_b();
}

void SI_SC_ComboEMFields_Cyl3D::
    SetupCPMLMagFields()
{
    m_MCPMLFields_Cyl3D = new SI_SC_CPML_MagFields_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_MCPMLFields_Cyl3D->SetDelTime(dt);

    m_MCPMLFields_Cyl3D->Setup();
    m_MCPMLFields_Cyl3D->Setup_PML_a_b();
}

void SI_SC_ComboEMFields_Cyl3D::
    SetupMurElecFields()
{
    m_EMurFields_Cyl3D = new SI_SC_Mur_ElecFieldsSet_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_EMurFields_Cyl3D->SetDelTime(dt);

    m_EMurFields_Cyl3D->Setup();
}

void SI_SC_ComboEMFields_Cyl3D::
    Write_PML_Inf()
{
    ostringstream sstr1;
    sstr1 << "CPML_E";
    sstr1 << ".txt";
    string oFileName1 = sstr1.str();
    ofstream txtStream1(oFileName1.c_str());

    m_ECPMLFields_Cyl3D->Write_PML_a_b(txtStream1);

    ostringstream sstr2;
    sstr2 << "CPML_M";
    sstr2 << ".txt";
    string oFileName2 = sstr2.str();
    ofstream txtStream2(oFileName2.c_str());

    m_MCPMLFields_Cyl3D->Write_PML_a_b(txtStream2);
}

vector<GridEdgeData *> &
SI_SC_ComboEMFields_Cyl3D::
    GetCntrElecEdges()
{
    return m_ECntrFields_Cyl3D->GetEdgeDatas();
}

vector<GridVertexData *> &
SI_SC_ComboEMFields_Cyl3D::
    GetCntrElecVertices()
{
    return m_ECntrFields_Cyl3D->GetVertexDatas();
}
