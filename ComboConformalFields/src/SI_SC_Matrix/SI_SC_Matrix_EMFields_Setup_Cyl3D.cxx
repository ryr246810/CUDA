#include <stdlib.h>
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include "SI_SC_ComboEMFields_Cyl3D.hxx"
#include "ComboFields_Dynamic_Srcs_Cyl3D.hxx"
#include "BaseFunctionDefine.hxx"

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    Setup()
{
    SetupCntrElecFields();
    SetupCntrMagFields();

    SetupCPMLElecFields();
    SetupCPMLMagFields();
    SetupMurElecFields();
    
    BuildDatas();
    BuildFuncs();
    
    Dynamic_ComboEMFieldsBase::Setup();
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetupCntrElecFields()
{
    set<Standard_Integer> theBndsDefine;
    GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet(theBndsDefine);

    m_ECntrFields_Cyl3D = new SI_SC_ElecFields_Cyl3D(GetFldsDefCntr(), EXCLUDING);
    m_ECntrFields_Cyl3D->SetMaterials(theBndsDefine);
    Standard_Real dt = this->GetDelTime();
    m_ECntrFields_Cyl3D->SetDelTime(dt);
    m_ECntrFields_Cyl3D->Setup();

    phi_num = m_ECntrFields_Cyl3D->GetGridGeom_Cyl3D()->GetDimPhi();
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
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

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetupCPMLElecFields()
{
    m_ECPMLFields_Cyl3D = new SI_SC_CPML_ElecFields_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_ECPMLFields_Cyl3D->SetDelTime(dt);
    m_ECPMLFields_Cyl3D->Setup();
    m_ECPMLFields_Cyl3D->Setup_PML_a_b();
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetupCPMLMagFields()
{
    m_MCPMLFields_Cyl3D = new SI_SC_CPML_MagFields_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_MCPMLFields_Cyl3D->SetDelTime(dt);
    m_MCPMLFields_Cyl3D->Setup();
    m_MCPMLFields_Cyl3D->Setup_PML_a_b();
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetupMurElecFields()
{
    m_EMurFields_Cyl3D = new SI_SC_Mur_ElecFieldsSet_Cyl3D(GetFldsDefCntr());
    Standard_Real dt = this->GetDelTime();
    m_EMurFields_Cyl3D->SetDelTime(dt);
    m_EMurFields_Cyl3D->Setup();
}

void 
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetDamping(const Standard_Real theDamping)
{
    m_Damping = theDamping;
}

void
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    SetOrder(const Standard_Integer theOrder)
{
    m_Order = theOrder;

    if(theOrder == 1){
        m_Step = 1;
        m_b = new Standard_Real[m_Step];
        m_bb = new Standard_Real[m_Step];

        m_b[0] = 1.0;
        m_bb[0] = 1.0;
    }
    else if(theOrder == 2){
        m_Step = 1;
        m_b = new Standard_Real[m_Step];
        m_bb = new Standard_Real[m_Step];
        m_b[0] = 1.0;
        m_bb[0] = 1.0;
    }
    else if(theOrder == 3){
        m_Step = 3;
        m_b = new Standard_Real[m_Step];
        m_bb = new Standard_Real[m_Step];

        m_bb[0] = 2.0 / 3.0;
        m_bb[1] = -2.0 / 3.0;
        m_bb[2] = 1.0;

        m_b[0] = 7.0 / 24.0;
        m_b[1] = 3.0 / 4.0;
        m_b[2] = -1.0 / 24.0;
    }
    else if(theOrder == 4){
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
    else{
        cout<<"error ------------SI_SC_ComboEMFields::SetOrder-------------should be not larger than 4"<<endl;
        exit(1);
    }
}

vector<GridEdgeData*>&
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    GetCntrElecEdges()
{
    return m_ECntrFields_Cyl3D->GetEdgeDatas();
}

vector<GridVertexData*>&
SI_SC_Matrix_EMFields_Cyl3D:: // no problem
    GetCntrElecVertices()
{
    return m_ECntrFields_Cyl3D->GetVertexDatas();
}