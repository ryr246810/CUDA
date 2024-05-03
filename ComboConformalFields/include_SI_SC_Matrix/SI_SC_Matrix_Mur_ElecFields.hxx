#ifndef _SI_SC_Matrix_Mur_ElecFields_HeaderFile
#define _SI_SC_Matrix_Mur_ElecFields_HeaderFile

#include "FieldsBase.hxx"
#include "GridGeometry.hxx"

class GridEdgeData;

class SI_SC_Matrix_Mur_ElecFields : public FieldsBase
{
public:
    SI_SC_Matrix_Mur_ElecFields();
    SI_SC_Matrix_Mur_ElecFields(const FieldsDefineCntr* theCntr,
        const PortData& therPort);
    
    virtual ~SI_SC_Matrix_Mur_ElecFields();

public:
    virtual void Setup();
    virtual bool IsPhysDataMemoryLocated() const {return true;};
    void SetPort(const PortData& thePort);
    void SetPhiIndex(Standard_Integer phi_index) {m_PhiIndex = phi_index;};

public:
    void ZeroPhysDatas();

private:
    void SetupDataEdgeDatas();
    void SetupDataSweptEdgeDatas();
    void SetupData();
    void SetupGridEdgeDatasEfficientLength();
    void SetupVP();

public:
    virtual void Advance_SI(const Standard_Real si_scale);
    virtual void Advance_SI_Damping(const Standard_Real si_scale,
        const Standard_Real damping_scale);
    virtual void Advance();

private:
    PortData m_MurPort;

    Standard_Real m_VBar;
    Standard_Real m_Step;
    Standard_Integer m_PhiIndex;

    vector<GridEdgeData*> m_MurPortEdgeDatas;
    vector<GridEdgeData*> m_FreeSpaceEdgeDatas;

    vector<GridVertexData*> m_MurPortSweptEdgeDatas;
    vector<GridVertexData*> M_FreeSpaceSweptEdgeDatas;
};


#endif