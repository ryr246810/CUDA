#ifndef _SI_SC_Matrix_EMFields_Cyl3D_HeaderFile
#define _SI_SC_Matrix_EMFields_Cyl3D_HeaderFile

#include "Dynamic_ComboEMFieldsBase.hxx"

#include "SI_SC_ElecFields_Cyl3D.hxx"
#include "SI_SC_MagFields_Cyl3D.hxx"
#include "SI_SC_CPML_MagFields_Cyl3D.hxx"
#include "SI_SC_CPML_ElecFields_Cyl3D.hxx"
#include "SI_SC_Mur_ElecFieldsSet_Cyl3D.hxx"

#include "SI_SC_Matrix_ElecFields_Cyl3D.hxx"
#include "SI_SC_Matrix_MagFields_Cyl3D.hxx"
#include "SI_SC_Matrix_CPML_MagFields_Cyl3D.hxx"
#include "SI_SC_Matrix_CPML_ElecFields_Cyl3D.hxx"
#include "SI_SC_Matrix_Mur_ElecFieldsSet_Cyl3D.hxx"

#include "SI_SC_Matrix_Elec_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Elec_CPML_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Elec_Mur_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Mag_Func_Cyl3D.cuh"
#include "SI_SC_Matrix_Mag_CPML_Func_Cyl3D.cuh"
#include <BaseFunctionDefine.hxx>

extern Standard_Real* m_h_d_MphiDatasPtr;
extern Standard_Real* m_h_d_MzrDatasPtr;
extern Standard_Real* m_h_d_EphiDatasPtr;
extern Standard_Real* m_h_d_EzrDatasPtr;

class SI_SC_Matrix_EMFields_Cyl3D: public Dynamic_ComboEMFieldsBase
{
public:
    SI_SC_Matrix_EMFields_Cyl3D();
    SI_SC_Matrix_EMFields_Cyl3D(const FieldsDefineCntr* _cntr);

    virtual ~SI_SC_Matrix_EMFields_Cyl3D();

public:
    virtual void Advance();
    virtual void AdvanceWithDamping();
    virtual void BuildCUDADatas();
    virtual void CleanCUDADatas();

    void AdvanceElecCntr();
    void AdvanceElecCPML();
    
    void AdvanceElecCntr_WithDamping(const Standard_Real damping_scale);
    void AdvanceElecCPML_WithDamping(const Standard_Real damping_scale);
    void AdvanceElecMur_WithDamping(const Standard_Real damping_scale);
    void AdvanceElecMurVoltage_WithDamping(const Standard_Real damping_scale);

    void AdvanceMagCntr();
    void AdvanceMagCPML();

    void AdvanceMagCntr_WithDamping();
    void AdvanceMagCPML_WithDamping();

    virtual void TransDgnData();
    virtual void InitMatrixData();

public:
    virtual void Setup();
    virtual bool IsPhysDataMemoryLocated();
    virtual void ZeroPhysDatas();

    virtual void SetOrder(const Standard_Integer theOrder);
    virtual void SetDamping(const Standard_Real theDamping);

    virtual vector<GridEdgeData*>& GetCntrElecEdges();
    virtual vector<GridVertexData*>& GetCntrElecVertices();

public:
    virtual void Write_PML_Inf();

    virtual void Get_Ezr_Info(Standard_Real** ptr, Standard_Size* size){
        *ptr  = m_EzrDatasPtr;
        *size = m_EzrDatasSize;
    };

    virtual void Get_Ephi_Info(Standard_Real** ptr, Standard_Size* size){
        *ptr  = m_EphiDatasPtr;
        *size = m_EphiDatasSize;
    };

    virtual void Get_Mzr_Info(Standard_Real** ptr, Standard_Size* size){
        *ptr  = m_MzrDatasPtr;
        *size = m_MzrDatasSize;
    };

    virtual void Get_Mphi_Info(Standard_Real** ptr, Standard_Size* size){
        *ptr  = m_MphiDatasPtr;
        *size = m_MphiDatasSize;
    };

    virtual void Get_cuda_ptr(Standard_Real** EzrPtr, Standard_Real** EphiPtr, Standard_Real** MzrPtr, Standard_Real** MphiPtr){
        *EzrPtr  = m_h_d_EzrDatasPtr;
        *EphiPtr = m_h_d_EphiDatasPtr;
        *MzrPtr  = m_h_d_MzrDatasPtr;
        *MphiPtr = m_h_d_MphiDatasPtr;
    };

private:
    void SetupCntrElecFields();
    void SetupCntrMagFields();
    void SetupCPMLElecFields();
    void SetupCPMLMagFields();
    void SetupMurElecFields();

private:
    void Build_Elec_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& dynJIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& dynHIndex);

    void Build_Elec_CPML_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& dynHIndex,
        const Standard_Integer& PE1,
        const Standard_Integer& PE2);

    void Build_Elec_Mur_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& preTStepEFldIndx);

    void Build_Elec_MurVoltage();

    void Build_Elec_MurVoltage_Func(const bool doDamping,
        const Standard_Real& dt,
        const Standard_Integer& dynEIndex,
        const Standard_Integer& PreIndex,
        const Standard_Integer& AEIndex,
        const Standard_Integer& BEIndex,
        const Standard_Integer& preTStepEFldIndx);
  
    void Build_Mag_Func(const Standard_Real& dt,
		const Standard_Integer& dynHIndex,
		const Standard_Integer& dynJIndex,
        const Standard_Integer& dynEIndex);
  
    void Build_Mag_CPML_Func(const Standard_Real& dt,
		const Standard_Integer& dynHIndex,
		const Standard_Integer& PM1Index,
        const Standard_Integer& PM2Index,
		const Standard_Integer& dynEIndex);

    void BuildDatas();
    void BuildMphiDatas();
    void BuildMzrDatas();
    void BuildEphiDatas();
    void BuildEzrDatas();
    void BuildEAxisDatas();

    void BuildCPMLEzrDatas();
    void BuildCPMLEphiDatas();
    void BuildCPMLMzrDatas();
    void BuildCPMLMphiDatas();
    void BuildCPMLEAxisDatas();

    //Build Data for GPU
    void BuildEdgeDatas();
    void BuildFaceDatas();
    void BuildVertexDatas();
    void BuildEdgeAxisDatas();
    void BuildNearEdgeDatas();
    void BuildCellDatas();

    void CleanEdgeDatas();
    void CleanFaceDatas();
    void CleanVertexDatas();
    void CleanEdgeAxisDatas();
    void CleanNearEdgeDatas();
    void CleanCellDatas();
    //Build Data for GPU
    
    void CleanDatas();
    void CleanMphiDatas();
    void CleanMzrDatas();
    void CleanEphiDatas();
    void CleanEzrDatas();
    void CleanEAxisDatas();

    void CleanCPMLEzrDatas();
    void CleanCPMLEphiDatas();
    void CleanCPMLMzrDatas();
    void CleanCPMLMphiDatas();
    void CleanCPMLEAxisDatas();

    void BuildFuncs();
    Standard_Integer Get_Phi_Num() {return phi_num;};

private:
    Standard_Integer m_Order;
    Standard_Integer m_Step;

    Standard_Real* m_b;
    Standard_Real* m_bb;

    Standard_Integer phi_num;
    Standard_Real m_VBAR;

private:
    Standard_Real m_Damping;
    Standard_Real m_ChiParam;
    Standard_Real m_ParabolicDampingTerm;

    SI_SC_ElecFields_Cyl3D* m_ECntrFields_Cyl3D;
    SI_SC_CPML_ElecFields_Cyl3D* m_ECPMLFields_Cyl3D;
    
    SI_SC_MagFields_Cyl3D* m_MCntrFields_Cyl3D;
    SI_SC_CPML_MagFields_Cyl3D* m_MCPMLFields_Cyl3D;
    SI_SC_Mur_ElecFieldsSet_Cyl3D* m_EMurFields_Cyl3D;

private:

    Cempic_Size  m_OneMphiPhysDataNum; 
    Cempic_Size  m_OneMzrPhysDataNum; 
    Cempic_Size  m_OneEphiPhysDataNum;
    Cempic_Size  m_OneEzrPhysDataNum;
    Cempic_Size  m_OneEAxisPhysDataNum;

    Cempic_Size  m_OneMphiPMLPhysDataNum; 
    Cempic_Size  m_OneMzrPMLPhysDataNum; 
    Cempic_Size  m_OneEphiPMLPhysDataNum;
    Cempic_Size  m_OneEzrPMLPhysDataNum;
    Cempic_Size  m_OneEAxisPMLPhysDataNum;

    Standard_Real* m_MphiDatasPtr;
    Standard_Real* m_MzrDatasPtr;
    Standard_Real* m_EphiDatasPtr;
    Standard_Real* m_EzrDatasPtr;
    Standard_Real* m_EAxisDatasPtr;
    Standard_Real* m_EzrOutDatasPtr;
    Standard_Real* m_EphiOutDatasPtr;
    Standard_Real* m_EphiNearDatasPtr;
    Standard_Real* m_CPMLENearDatasPtr;
    Standard_Real* m_EAxisDualContourValue;
    Standard_Real* m_EAxisPhysDataJ;
    Standard_Real* m_EAxisPhysDataE;
    Standard_Real* m_CPMLEAxisDualContourValue;
    Standard_Real* m_CPMLEAxisPhysDataPE1;
    Standard_Real* m_CPMLEAxisPhysDataE;

    Standard_Real* m_CPML_MphiDatasPtr;
    Standard_Real* m_CPML_MzrDatasPtr;
    Standard_Real* m_CPML_EphiDatasPtr;
    Standard_Real* m_CPML_EzrDatasPtr;
    Standard_Real* m_CPML_EAxisDatasPtr;

    Cempic_Size  m_MphiDatasNum;
    Cempic_Size  m_MzrDatasNum;
    Cempic_Size  m_EphiDatasNum;
    Cempic_Size  m_EzrDatasNum;
    Cempic_Size  m_EAxisDatasNum;

    Cempic_Size  m_CPML_MphiDatasNum;
    Cempic_Size  m_CPML_MzrDatasNum;
    Cempic_Size  m_CPML_EphiDatasNum;
    Cempic_Size  m_CPML_EzrDatasNum;
    Cempic_Size  m_CPML_EAxisDatasNum;
    Cempic_Size  m_Mur_EzrDatasNum;
    Cempic_Size  m_Mur_EphiDatasNum;
    Cempic_Size  m_MurVoltage_EzrDatasNum;
    Cempic_Size  m_MurVoltage_EphiDatasNum;

    Cempic_Size  m_MphiDatasSize;
	Cempic_Size  m_MzrDatasSize;
	Cempic_Size  m_EphiDatasSize;
	Cempic_Size  m_EzrDatasSize;
    Cempic_Size  m_EAxisDatasSize;
    Cempic_Size  m_EzrOutDatasSize;
    Cempic_Size  m_EphiOutDatasSize;
    Cempic_Size  m_EphiNearDatasSize;
    Cempic_Size  m_CPMLENearDatasSize;

    Standard_Integer bytesMphiData;
    Standard_Integer bytesMzrData;
    Standard_Integer bytesEphiData;
    Standard_Integer bytesEzrData;  
    Standard_Integer bytesdamping_scale; 
    Standard_Integer bytesParameters;
    Standard_Integer bytesEAxisData;
    Standard_Integer bytesCPMLEAxisData;
    Standard_Integer bytesEzrOutData;
    Standard_Integer bytesEphiOutData;
    Standard_Integer bytesEphiNearData;
    Standard_Integer bytesCPMLENearData;

    // CUDA Vars 新空间指针
    // Standard_Real* m_h_d_MphiDatasPtr;
    // Standard_Real* m_h_d_MzrDatasPtr;
    // Standard_Real* m_h_d_EphiDatasPtr;
    // Standard_Real* m_h_d_EzrDatasPtr;
    Standard_Real* m_h_d_EAxisDatasPtr;
    Standard_Real* m_h_d_EzrOutDatasPtr;
    Standard_Real* m_h_d_EphiOutDatasPtr;
    Standard_Real* m_h_d_EphiNearDatasPtr;
    Standard_Real* m_h_d_CPMLENearDatasPtr;
    Standard_Real* m_h_d_EAxisDualContourValue;
    Standard_Real* m_h_d_EAxisPhysDataJ;
    Standard_Real* m_h_d_EAxisPhysDataE;
    Standard_Real* m_h_d_CPMLEAxisDualContourValue;
    Standard_Real* m_h_d_CPMLEAxisPhysDataPE1;
    Standard_Real* m_h_d_CPMLEAxisPhysDataE;

    Standard_Real* m_h_d_CPMLMphiDatasPtr;
    Standard_Real* m_h_d_CPMLMzrDatasPtr;
    Standard_Real* m_h_d_CPMLEphiDatasPtr;
    Standard_Real* m_h_d_CPMLEzrDatasPtr;
    Standard_Real* m_h_d_CPMLEAxisDatasPtr;

    Standard_Real* m_h_d_damping_scale;
    Standard_Integer * m_h_d_Parameters;

private:
    FIT_Elec_Func_Cyl3D*     m_EphiFuncArray_Cyl3D;
    FIT_Elec_Func_Cyl3D*     m_EzrFuncArray_Cyl3D;
    FIT_Elec_Func_Cyl3D*     m_EAxisFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D* m_EphiPMLFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D* m_EzrPMLFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D* m_EAxisPMLFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D* m_EzrMurFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D* m_EphiMurFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D* m_EzrMurVoltageFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D* m_EphiMurVoltageFuncArray_Cyl3D;
    
    FIT_Mag_Func_Cyl3D*     m_MphiFuncArray_Cyl3D;
    FIT_Mag_Func_Cyl3D*     m_MzrFuncArray_Cyl3D;
    FIT_Mag_PML_Func_Cyl3D* m_MphiPMLFuncArray_Cyl3D;
    FIT_Mag_PML_Func_Cyl3D* m_MzrPMLFuncArray_Cyl3D;

    // CUDA Vars
    FIT_Mag_Func_Cyl3D*         m_h_d_MphiFuncArray_Cyl3D;
    FIT_Mag_Func_Cyl3D*         m_h_d_MzrFuncArray_Cyl3D;
    FIT_Mag_PML_Func_Cyl3D*     m_h_d_CPMLMphiFuncArray_Cyl3D;
    FIT_Mag_PML_Func_Cyl3D*     m_h_d_CPMLMzrFuncArray_Cyl3D;

    FIT_Elec_Func_Cyl3D*        m_h_d_EphiFuncArray_Cyl3D;
    FIT_Elec_Func_Cyl3D*        m_h_d_EzrFuncArray_Cyl3D;
    FIT_Elec_Func_Cyl3D*        m_h_d_EAxisFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D*    m_h_d_CPMLEphiFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D*    m_h_d_CPMLEzrFuncArray_Cyl3D;
    FIT_Elec_PML_Func_Cyl3D*    m_h_d_CPMLEAxisFuncArray_Cyl3D;

    FIT_Elec_Mur_Func_Cyl3D*    m_h_d_MurEphiFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D*    m_h_d_MurEzrFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D*    m_h_d_MurVoltageEphiFuncArray_Cyl3D;
    FIT_Elec_Mur_Func_Cyl3D*    m_h_d_MurVoltageEzrFuncArray_Cyl3D;

    Standard_Real Ebar;
    Standard_Real Ebar2;

    Standard_Real* amp;
    Standard_Integer amp_size;

    vector<GridEdgeData*> m_MurEdgeDatas;
	vector<GridEdgeData*> m_FreeEdgeDatas;
	vector<GridVertexData*> m_MurSweptEdgeDatas;
	vector<GridVertexData*> m_FreeSweptEdgeDatas;
};




#endif