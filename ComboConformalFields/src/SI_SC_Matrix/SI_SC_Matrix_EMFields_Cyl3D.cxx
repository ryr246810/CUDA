#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"
#include <stdlib.h>

#include "ComboFields_Dynamic_Srcs_Cyl3D.hxx"

#include "CUDAHeader.cuh"

SI_SC_Matrix_EMFields_Cyl3D::
SI_SC_Matrix_EMFields_Cyl3D()
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

    m_MphiDatasPtr = NULL;
    m_MzrDatasPtr = NULL;
    m_EphiDatasPtr = NULL;
    m_EzrDatasPtr = NULL;

    m_CPML_MphiDatasPtr = NULL;
    m_CPML_MzrDatasPtr = NULL;
    m_CPML_EphiDatasPtr = NULL;
    m_CPML_EzrDatasPtr = NULL;

    m_EphiFuncArray_Cyl3D = NULL;
    m_EzrFuncArray_Cyl3D = NULL;
    m_MphiFuncArray_Cyl3D = NULL;
    m_MzrFuncArray_Cyl3D = NULL;

    m_EphiPMLFuncArray_Cyl3D = NULL;
    m_EzrPMLFuncArray_Cyl3D = NULL;
    m_MphiPMLFuncArray_Cyl3D = NULL;
    m_MzrPMLFuncArray_Cyl3D = NULL;

    m_OneMphiPhysDataNum = 0; 
    m_OneMzrPhysDataNum = 0; 
    m_OneEphiPhysDataNum = 0;
    m_OneEzrPhysDataNum = 0;

    m_OneMphiPMLPhysDataNum = 0; 
    m_OneMzrPMLPhysDataNum = 0; 
    m_OneEphiPMLPhysDataNum = 0;
    m_OneEzrPMLPhysDataNum = 0;

    m_MphiDatasNum = 0;
    m_MzrDatasNum = 0;
    m_EphiDatasNum = 0;
    m_EzrDatasNum = 0;

    m_CPML_MphiDatasNum = 0;
    m_CPML_MzrDatasNum = 0;
    m_CPML_EphiDatasNum = 0;
    m_CPML_EzrDatasNum = 0;

    // CUDA Vars
    // m_h_d_MphiDatasPtr  = NULL;
    // m_h_d_MzrDatasPtr   = NULL;
    // m_h_d_EphiDatasPtr  = NULL;
    // m_h_d_EzrDatasPtr   = NULL;
    // m_h_d_EAxisDatasPtr = NULL;

    // m_h_d_CPMLMphiDatasPtr  = NULL;
    // m_h_d_CPMLMzrDatasPtr   = NULL;
    // m_h_d_CPMLEphiDatasPtr  = NULL;
    // m_h_d_CPMLEzrDatasPtr   = NULL;
    // m_h_d_CPMLEAxisDatasPtr = NULL;

    // m_h_d_MphiFuncArray_Cyl3D  = NULL;
    // m_h_d_MzrFuncArray_Cyl3D   = NULL;
    // m_h_d_EphiFuncArray_Cyl3D  = NULL;
    // m_h_d_EzrFuncArray_Cyl3D   = NULL;
    // m_h_d_EAxisFuncArray_Cyl3D = NULL;

    // m_h_d_CPMLMphiFuncArray_Cyl3D  = NULL;
    // m_h_d_CPMLMzrFuncArray_Cyl3D   = NULL;
    // m_h_d_CPMLEphiFuncArray_Cyl3D  = NULL;
    // m_h_d_CPMLEzrFuncArray_Cyl3D   = NULL;
    // m_h_d_CPMLEAxisFuncArray_Cyl3D = NULL;
}

SI_SC_Matrix_EMFields_Cyl3D::
~SI_SC_Matrix_EMFields_Cyl3D()
{
    CleanDatas();

    // printf("----cut---\n\n");

    if(m_ECntrFields_Cyl3D != NULL) delete m_ECntrFields_Cyl3D;
    if(m_MCntrFields_Cyl3D != NULL) delete m_MCntrFields_Cyl3D;

    if(m_ECPMLFields_Cyl3D != NULL) delete m_ECPMLFields_Cyl3D;
    if(m_MCPMLFields_Cyl3D != NULL) delete m_MCPMLFields_Cyl3D;
    if(m_EMurFields_Cyl3D != NULL) delete m_EMurFields_Cyl3D;

    if(m_b !=NULL ) delete[] m_b;
    if(m_bb !=NULL ) delete[] m_bb;
    
    if(m_EphiFuncArray_Cyl3D != NULL) delete [] m_EphiFuncArray_Cyl3D;
    if(m_EzrFuncArray_Cyl3D != NULL) delete [] m_EzrFuncArray_Cyl3D;
    if(m_EAxisFuncArray_Cyl3D != NULL) delete [] m_EAxisFuncArray_Cyl3D;
    if(m_MphiFuncArray_Cyl3D != NULL) delete [] m_MphiFuncArray_Cyl3D;
    if(m_MzrFuncArray_Cyl3D != NULL) delete [] m_MzrFuncArray_Cyl3D;

    if(m_EphiPMLFuncArray_Cyl3D != NULL) delete [] m_EphiPMLFuncArray_Cyl3D;
    if(m_EzrPMLFuncArray_Cyl3D != NULL) delete [] m_EzrPMLFuncArray_Cyl3D;
    if(m_EAxisPMLFuncArray_Cyl3D != NULL) delete [] m_EAxisPMLFuncArray_Cyl3D;
    if(m_MphiPMLFuncArray_Cyl3D != NULL) delete [] m_MphiPMLFuncArray_Cyl3D;
    if(m_MzrPMLFuncArray_Cyl3D != NULL) delete [] m_MzrPMLFuncArray_Cyl3D;

    // if (m_MphiDatasPtr != NULL) aligned_free(m_MphiDatasPtr);
    // if (m_MzrDatasPtr != NULL) aligned_free(m_MzrDatasPtr);
    // if (m_EphiDatasPtr != NULL) aligned_free(m_EphiDatasPtr);
    // if (m_EzrDatasPtr != NULL) aligned_free(m_EzrDatasPtr);

    // if (m_CPML_MphiDatasPtr != NULL) aligned_free(m_CPML_MphiDatasPtr);
    // if (m_CPML_MzrDatasPtr != NULL) aligned_free(m_CPML_MzrDatasPtr);
    // if (m_CPML_EphiDatasPtr != NULL) aligned_free(m_CPML_EphiDatasPtr);
    // if (m_CPML_EzrDatasPtr != NULL) aligned_free(m_CPML_EzrDatasPtr);

    // if(m_h_d_MphiDatasPtr != NULL) aligned_free(m_h_d_MphiDatasPtr);
    // if(m_h_d_MzrDatasPtr != NULL) aligned_free(m_h_d_MzrDatasPtr);
    // if(m_h_d_EphiDatasPtr != NULL) aligned_free(m_h_d_EphiDatasPtr);
    // if(m_h_d_EzrDatasPtr != NULL) aligned_free(m_h_d_EzrDatasPtr);
    // if(m_h_d_EAxisDatasPtr != NULL) aligned_free(m_h_d_EAxisDatasPtr);

    // printf("----cut---\n\n");

    #ifdef __cuda__
        if(m_h_d_MphiFuncArray_Cyl3D != NULL) cudaFree(m_h_d_MphiFuncArray_Cyl3D);
        if(m_h_d_MzrFuncArray_Cyl3D != NULL) cudaFree(m_h_d_MzrFuncArray_Cyl3D);
        if(m_h_d_EphiFuncArray_Cyl3D != NULL) cudaFree(m_h_d_EphiFuncArray_Cyl3D);
        if(m_h_d_EzrFuncArray_Cyl3D != NULL) cudaFree(m_h_d_EzrFuncArray_Cyl3D);
        if(m_h_d_EAxisFuncArray_Cyl3D != NULL) cudaFree(m_h_d_EAxisFuncArray_Cyl3D);

        if(m_h_d_CPMLMphiFuncArray_Cyl3D != NULL) cudaFree(m_h_d_CPMLMphiFuncArray_Cyl3D);
        if(m_h_d_CPMLMzrFuncArray_Cyl3D != NULL) cudaFree(m_h_d_CPMLMzrFuncArray_Cyl3D);
        if(m_h_d_CPMLEphiFuncArray_Cyl3D != NULL) cudaFree(m_h_d_CPMLEphiFuncArray_Cyl3D);
        if(m_h_d_CPMLEzrFuncArray_Cyl3D != NULL) cudaFree(m_h_d_CPMLEzrFuncArray_Cyl3D);
        if(m_h_d_CPMLEAxisFuncArray_Cyl3D != NULL) cudaFree(m_h_d_CPMLEAxisFuncArray_Cyl3D);
    #endif
}

bool
SI_SC_Matrix_EMFields_Cyl3D::
IsPhysDataMemoryLocated()
{
    return true;
}

void
SI_SC_Matrix_EMFields_Cyl3D::
ZeroPhysDatas()
{
    m_ECntrFields_Cyl3D->ZeroPhysDatas();
    m_MCntrFields_Cyl3D->ZeroPhysDatas();
    m_ECPMLFields_Cyl3D->ZeroPhysDatas();
    m_MCPMLFields_Cyl3D->ZeroPhysDatas();

    m_EMurFields_Cyl3D->ZeroPhysDatas();
}

void
SI_SC_Matrix_EMFields_Cyl3D::
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