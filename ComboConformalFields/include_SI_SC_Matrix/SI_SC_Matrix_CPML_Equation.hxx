#ifndef _SI_SC_Matrix_CPML_Equation_HeaderFile
#define _SI_SC_Matirx_CPML_Equation_HeaderFile

#include "DataBase.hxx"
#include "BaseFunctionDefine.hxx"
#include "GridVertexData.hxx"
#include "GridEdgeData.hxx"
#include "GridFaceData.cuh"

void Matrix_Compute_a_b_SI_SC(DataBase* theData,
    const Standard_Integer theTruncDir,
    const Standard_Real Dt,
    Standard_Real& a,
    Standard_Real& b);

void Matrix_Compute_Dual_a_b_SI_SC(DataBase* theData,
    const Standard_Integer theTruncDir,
    const Standard_Real Dt,
    Standard_Real& a,
    Standard_Real& b);

void Matrix_Get_a_b_SI_SC(DataBase* theData,
    const Standard_Integer theTruncDir,
    const Standard_Real Dt,
    Standard_Real& a,
    Standard_Real& b);

#endif




