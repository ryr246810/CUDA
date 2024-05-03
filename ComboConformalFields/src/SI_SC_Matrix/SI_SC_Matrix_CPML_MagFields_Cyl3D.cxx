#include "SI_SC_Matrix_CPML_MagFields_Cyl3D.hxx"

#include "GridFaceData.cuh"
#include "GridEdgeData.hxx"
#include "GridFace.hxx"
#include "GridEdge.hxx"

#include "SI_SC_CPML_Equation.hxx"

SI_SC_Matrix_CPML_MagFields_Cyl3D::
SI_SC_Matrix_CPML_MagFields_Cyl3D()
    :SI_SC_Matrix_MagFields_Cyl3D()
{

}

SI_SC_Matrix_CPML_MagFields_Cyl3D::
SI_SC_Matrix_CPML_MagFields_Cyl3D(const FieldsDefineCntr* theCntr)
    :SI_SC_Matrix_MagFields_Cyl3D(theCntr, INCLUDING)
{

}

SI_SC_Matrix_CPML_MagFields_Cyl3D::
~SI_SC_Matrix_CPML_MagFields_Cyl3D()
{

}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Setup()
{
    Standard_Real dt = this->GetDelTime();

    m_FaceMagFlds_Cyl3D = new FaceMagFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
    m_FaceMagFlds_Cyl3D->clearMaterials();
    m_FaceMagFlds_Cyl3D->AppendingMaterial(PML);
    m_FaceMagFlds_Cyl3D->SetDelTime(dt);
    m_FaceMagFlds_Cyl3D->Setup();

    m_SweptFaceMagFlds_Cyl3D = new SweptFaceMagFldsBase_Cyl3D(GetFldsDefCntr(), m_Rule);
    m_SweptFaceMagFlds_Cyl3D->clearMaterials();
    m_SweptFaceMagFlds_Cyl3D->AppendingMaterial(PML);
    m_SweptFaceMagFlds_Cyl3D->SetDelTime(dt);
    m_SweptFaceMagFlds_Cyl3D->Setup();
}

bool
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    IsPhysDataMemoryLocated() const
{
    return SI_SC_Matrix_MagFields_Cyl3D::IsPhysDataMemoryLocated();
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    ZeroPhysDatas()
{
    vector<GridEdgeData*> &theSweptFaceDatas = m_SweptFaceMagFlds_Cyl3D->GetDatas();
    Standard_Size ne = theSweptFaceDatas.size();
    for(Standard_Size i = 0; i < ne; ++i){
        theSweptFaceDatas[i]->ZeroSweptPhysDatas();
    }

    vector<GridFaceData*> &theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
    Standard_Size nf = theFaceDatas.size();
    for(Standard_Size i = 0; i < nf; ++i){
        theFaceDatas[i]->ZeroPhysDatas();
    }
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Advance()
{
    DynObj::Advance();
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Advance_SI(const Standard_Real si_scale)
{
    Standard_Integer TemporalEFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
    Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

    Standard_Integer thePM1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM1_PhysDataIndex();
    Standard_Integer thePM2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM2_PhysDataIndex();

    Standard_Real dt = GetDelTime();

    vector<GridFaceData*> &theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
    vector<GridEdgeData*> &theSweptFaceData = m_SweptFaceMagFlds_Cyl3D->GetDatas();

    Advance_CPML_MagElems_SI_SC(theFaceDatas,
                                dt, si_scale,
                                TemporalHFieldIndex, thePM1Index, thePM2Index,
                                TemporalEFieldIndex);

    Advance_CPML_MagElems_SI_SC(theSweptFaceData,
                                dt, si_scale,
                                TemporalHFieldIndex, thePM1Index, thePM2Index,
                                TemporalEFieldIndex);
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Advance_SI_Damping(const Standard_Real si_scale)
{
    Standard_Integer TemporalHFieldIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

    Standard_Integer thePM1Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM1_PhysDataIndex();
    Standard_Integer thePM2Index = GetFldsDefCntr()->GetFieldsDefineRules()->Get_CPML_PM2_PhysDataIndex();

    Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();

    Standard_Real dt = GetDelTime();

    vector<GridFaceData*> &theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
    vector<GridEdgeData*> &theSweptFaceData = m_SweptFaceMagFlds_Cyl3D->GetDatas();

    Advance_CPML_MagElems_SI_SC(theFaceDatas,
                                dt, si_scale,
                                TemporalHFieldIndex, thePM1Index, thePM2Index,
                                BEIndex);

    Advance_CPML_MagElems_SI_SC(theSweptFaceData,
                                dt, si_scale,
                                TemporalHFieldIndex, thePM1Index, thePM2Index,
                                BEIndex);
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Setup_PML_a_b()
{
    Standard_Real a = 0.;
    Standard_Real b = 0.;
    Standard_Real dt = GetDelTime();

    vector<GridEdgeData*> &theSweptFaceDatas = m_SweptFaceMagFlds_Cyl3D->GetDatas();
    for(Standard_Integer i = 0; i < theSweptFaceDatas.size(); ++i){
        for(Standard_Integer dir = 0; dir < 2; ++dir){
            Compute_a_b_SI_SC(theSweptFaceDatas[i], dir, dt, a, b);

            theSweptFaceDatas[i]->SetPML_a(dir, a);
            theSweptFaceDatas[i]->SetPML_b(dir, b);
        }
    }

    vector<GridFaceData*> &theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
    for(Standard_Integer i = 0; i < theFaceDatas.size(); ++i){
        for(Standard_Integer dir = 0; dir < 2; ++dir){
            Compute_a_b_SI_SC(theFaceDatas[i], dir, dt, a, b);

            theFaceDatas[i]->SetPML_a(dir, a);
            theFaceDatas[i]->SetPML_b(dir, b);
        }
    }
}

void
SI_SC_Matrix_CPML_MagFields_Cyl3D::
    Write_PML_a_b(std::ostream &theoutstream) const
{
    vector<GridEdgeData*> &theSweptFaceDatas = m_SweptFaceMagFlds_Cyl3D->GetDatas();
    for(Standard_Integer i = 0; i < theSweptFaceDatas.size(); ++i){
        theoutstream << " sigma=(";
        theoutstream << theSweptFaceDatas[i]->GetPMLSigma(0) << "," << theSweptFaceDatas[i]->GetPMLSigma(1) << ")";

        theoutstream << " alpha=(";
        theoutstream << theSweptFaceDatas[i]->GetPMLAlpha(0) << "," << theSweptFaceDatas[i]->GetPMLAlpha(1) << ")";

        theoutstream << " kappa=(";
        theoutstream << theSweptFaceDatas[i]->GetPMLKappa(0) << "," << theSweptFaceDatas[i]->GetPMLKappa(1) << ")";

        theoutstream << " a=(";
        theoutstream << theSweptFaceDatas[i]->GetPML_a(0) << "," << theSweptFaceDatas[i]->GetPML_a(1) << ")" ;

        theoutstream << " b=(";
        theoutstream << theSweptFaceDatas[i]->GetPML_b(0) << "," << theSweptFaceDatas[i]->GetPML_b(1) << ")";
        theoutstream << endl;
    }

    vector<GridFaceData*> &theFaceDatas = m_FaceMagFlds_Cyl3D->GetDatas();
    for(Standard_Integer i = 0; i < theFaceDatas.size(); ++i){
        theoutstream << " sigma=(";
        theoutstream << theFaceDatas[i]->GetPMLSigma(0) << "," << theFaceDatas[i]->GetPMLSigma(1) << ")";

        theoutstream << " alpha=(";
        theoutstream << theFaceDatas[i]->GetPMLAlpha(0) << "," << theFaceDatas[i]->GetPMLAlpha(1) << ")";

        theoutstream << " kappa=(";
        theoutstream << theFaceDatas[i]->GetPMLKappa(0) << "," << theFaceDatas[i]->GetPMLKappa(1) << ")";

        theoutstream << " a=(";
        theoutstream << theFaceDatas[i]->GetPML_a(0) << "," << theFaceDatas[i]->GetPML_a(1) << ")" ;

        theoutstream << " b=(";
        theoutstream << theFaceDatas[i]->GetPML_b(0) << "," << theFaceDatas[i]->GetPML_b(1) << ")";
        theoutstream << endl;
    }
}