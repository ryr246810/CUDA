#include "Species_Cyl3D.cuh"
#include "Species_Cyl3D_SubFunc.cuh"
#include "CUDAHeader.cuh"

int Species_Cyl3D::Cuda_Init_Cuda_Constant_Vars(){
	const TxVector2D<Standard_Real> & orgs = global_grid->GetOrg();
	const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > & lVectors = global_grid->GetLVectors();
	const map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > & dlVectors = global_grid->GetDLVectors();
	const Standard_Real * minSteps = global_grid->GetMinSteps();
	Standard_Real minStep = global_grid->GetMinStep();
	const Standard_Integer * dimensions = global_grid->GetDimensions();
	const Standard_Integer phi_number = geom->GetDimPhi();

	checkCudaStatus(Cuda_Constant_Vars_Init(orgs, lVectors, dlVectors, minSteps, dimensions, phi_number, minStep, chargeOverMass));

	return 0;
}