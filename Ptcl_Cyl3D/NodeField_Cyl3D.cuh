#ifndef NODEFIELD_Cyl3D
#define NODEFIELD_Cyl3D

#include <TxMakerMap.h>
#include <ZRGrid.hxx>
#include <GridGeometry.hxx>
#include <PhysConsts.hxx>
#include <ComboFieldsDefineRules.hxx>
#include <IndexAndWeights_Cyl3D.cuh>
#include <SI_SC_ComboEMFields_Cyl3D.hxx>
#include <Dynamic_ComboEMFieldsBase.hxx>
#include <STFunc.hxx>
#include <math_Array2D.hxx>
#include <math_Array.hxx>
#include <BaseFunctionDefine.hxx>

extern TxVector<double>* d_E_node;
extern TxVector<double>* d_B_node;

class T_Node_Cyl3D{
public:
	int flag;
	DataBase* edge_data[2][2]; // 开辟一块 2*2 大小的数组，每个数组存一个 DataBase* 类型指针 T_Element 
	DataBase* minus_data[2][2];
	DataBase* face_data[4]; // GridFaceData 类
	DataBase* vertex_data[2]; // GridVertexData 类
	
	T_Node_Cyl3D(){
		flag = 0;
		for(int i = 0; i < 4; i++){
			*(edge_data[0] + i) = NULL;
			*(minus_data[0] + i) = NULL;
			face_data[i] = NULL;
		}

		for(int i = 0; i < 2; i++){
			vertex_data[i] = NULL;
		}
	}
};

class T_Node_Cyl3D_Info
{
public:
// 无参构造
	T_Node_Cyl3D_Info(){
		// 开辟的空间初始化置零
		memset(num, 0, sizeof(Standard_Real) * 3);
		memset(num1, 0, sizeof(Standard_Real) * 3);
		memset(tmpDataElec, 0, sizeof(Standard_Real) * 3);
		memset(tmpDataElec1, 0, sizeof(Standard_Real) * 3);
		memset(tmpDataMag, 0, sizeof(Standard_Real) * 3);
		memset(tmpDataMag1, 0, sizeof(Standard_Real) * 3);
		memset(m_MagEdge0, 0, sizeof(Standard_Real) * 2);
		memset(m_MagEdge1, 0, sizeof(Standard_Real) * 2);
		memset(m_MagMinus0, 0, sizeof(Standard_Real) * 2);
		memset(m_MagMinus1, 0, sizeof(Standard_Real) * 2);
		memset(m_MagFace, 0, sizeof(Standard_Real) * 4);
		memset(m_MagFace1, 0, sizeof(Standard_Real) * 4);
		memset(m_ElecEdge0, 0, sizeof(Standard_Real) * 2);
		memset(m_ElecEdge1, 0, sizeof(Standard_Real) * 2);
		memset(m_ElecVertex, 0, sizeof(Standard_Real) * 2);

		memset(m_MagEdge0_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_MagEdge1_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_MagMinus0_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_MagMinus1_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_ElecEdge0_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_ElecEdge1_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
		memset(m_MagFace_Ptr_Offset, 0, sizeof(Standard_Integer) * 4);
		memset(m_MagFace1_Ptr_Offset, 0, sizeof(Standard_Integer) * 4);
		memset(m_ElecVertex_Ptr_Offset, 0, sizeof(Standard_Integer) * 2);
	};

	~T_Node_Cyl3D_Info(){
		// for(int i = 0; i < 2; ++i){
		// 	if(m_MagEdge0[i] != NULL){
		// 		delete m_MagEdge0[i];
		// 		m_MagEdge0[i] = NULL;
		// 	}

		// 	if(m_MagEdge1[i] != NULL){
		// 		delete m_MagEdge1[i];
		// 		m_MagEdge1[i] = NULL;
		// 	}

		// 	if(m_MagMinus0[i] != NULL){
		// 		delete m_MagMinus0[i];
		// 		m_MagMinus0[i] = NULL;
		// 	}

		// 	if(m_MagMinus1[i] != NULL){
		// 		delete m_MagMinus1[i];
		// 		m_MagMinus1[i] = NULL;
		// 	}

		// 	if(m_ElecEdge0[i] != NULL){
		// 		delete m_ElecEdge0[i];
		// 		m_ElecEdge0[i] = NULL;
		// 	}

		// 	if(m_ElecEdge1[i] != NULL){
		// 		delete m_ElecEdge1[i];
		// 		m_ElecEdge1[i] = NULL;
		// 	}
		// }

		// for(int i = 0; i < 4; ++i){
		// 	if(m_MagFace[i] != NULL){
		// 		delete m_MagFace[i];
		// 		m_MagFace[i] = NULL;
		// 	}
		// }

		// for(int i = 0; i < 4; ++i){
		// 	if(m_MagFace1[i] != NULL){
		// 		delete m_MagFace1[i];
		// 		m_MagFace1[i] = NULL;
		// 	}
		// }

		// for(int i = 0; i < 2; ++i){
		// 	if(m_ElecVertex[i] != NULL){
		// 		delete m_ElecVertex[i];
		// 		m_ElecVertex[i] = NULL;
		// 	}
		// }
	};

public:
	Standard_Real num[3];
	Standard_Real num1[3];
	Standard_Real tmpDataElec[3];
	Standard_Real tmpDataElec1[3];
	Standard_Real tmpDataMag[3];
	Standard_Real tmpDataMag1[3];

	Standard_Real* m_MagEdge0[2];
	Standard_Integer m_MagEdge0_Ptr_Offset[2]; // Mzr

	Standard_Real* m_MagMinus0[2];
	Standard_Integer m_MagMinus0_Ptr_Offset[2]; // Mzr ???

	Standard_Real* m_MagEdge1[2];
	Standard_Integer m_MagEdge1_Ptr_Offset[2]; // Mzr

	Standard_Real* m_MagMinus1[2];
	Standard_Integer m_MagMinus1_Ptr_Offset[2]; // Mzr ???

	Standard_Real* m_MagFace[4]; 
	Standard_Integer m_MagFace_Ptr_Offset[4]; // Mphi

	Standard_Real* m_MagFace1[4]; 
	Standard_Integer m_MagFace1_Ptr_Offset[4]; // Mphi

	Standard_Real* m_ElecEdge0[2];
	Standard_Integer m_ElecEdge0_Ptr_Offset[2]; // Ezr

	Standard_Real* m_ElecEdge1[2];
	Standard_Integer m_ElecEdge1_Ptr_Offset[2]; // Ezr

	Standard_Real* m_ElecVertex[2];
	Standard_Integer m_ElecVertex_Ptr_Offset[2]; // Ephi

public:
	void CheckDatas(){
		for(int i = 0; i < 2; ++i){
			if(m_MagEdge0[i] == NULL){
				m_MagEdge0[i] = new Standard_Real;
				*m_MagEdge0[i] = 0;
			}

			if(m_MagEdge1[i] ==NULL){
				m_MagEdge1[i] = new Standard_Real;
				*m_MagEdge1[i] = 0;
			}

			if(m_MagMinus0[i] == NULL){
				m_MagMinus0[i] = new Standard_Real;
				*m_MagMinus0[i] = 0;
			}

			if(m_MagMinus1[i] == NULL){
				m_MagMinus1[i] = new Standard_Real;
				*m_MagMinus1[i] = 0;
			}

			if(m_ElecEdge0[i] == NULL){
				m_ElecEdge0[i] = new Standard_Real;
				*m_ElecEdge0[i] = 0;
			}

			if(m_ElecEdge1[i] == NULL){
				m_ElecEdge1[i] = new Standard_Real;
				*m_ElecEdge1[i] = 0;
			}
		}

		for(int i = 0; i < 4; ++i){
			if(m_MagFace[i] == NULL){
				m_MagFace[i] = new Standard_Real;
				*m_MagFace[i] = 0;
			}
		}

		for(int i = 0; i < 4; ++i){
			if(m_MagFace1[i] == NULL){
				m_MagFace1[i] = new Standard_Real;
				*m_MagFace1[i] = 0;
			}
		}

		for(int i = 0; i < 2; ++i){
			if(m_ElecVertex[i] == NULL){
				m_ElecVertex[i] = new Standard_Real;
				*m_ElecVertex[i] = 0;
			}
		}
	};
};

class Conformal_Current_Cuda{
public:
	int z;
	int r;
	int phi;
	int offset;
};

class NodeField_Cyl3D{
public:
	const ZRGrid * global_grid;
	GridGeometry_Cyl3D * gridGeom_Cyl3D;

	vector<double *> edges_z_current;
	vector<double *> Jz_to_edges_z;
	vector<double *> edges_r_current;
	vector<double *> Jr_to_edges_r;
	vector<double *> vertices_current;
	vector<double *> Jphi_to_vertices;
	
	int dynElecIndex;
	int dynMagIndex;
	int currentIndex;	

	int n_cell_z, n_cell_r,m_phi_number;

	T_Node_Cyl3D_Info* m_h_d_node_Conformal_info;

	Standard_Real* m_Ezr_Ptr;
	Standard_Real* m_Ephi_Ptr;
	Standard_Real* m_Mzr_Ptr;
	Standard_Real* m_Mphi_Ptr;

	Standard_Size  m_Ezr_dataSize;
	Standard_Size  m_Ephi_dataSize;
	Standard_Size  m_Mzr_dataSize;
	Standard_Size  m_Mphi_dataSize;

	TxVector<double>* dev_E_node_;
	TxVector<double>* dev_B_node_;
	TxVector<double>* dev_E_node_pre_;
	TxVector<double>* dev_E_node_curr_;

	double* dev_Jz;
	double* dev_Jr;
	double* dev_Jphi;
	
	Conformal_Current_Cuda* h_Jz_Current_Cuda;
	Conformal_Current_Cuda* h_Jr_Current_Cuda;
	Conformal_Current_Cuda* h_Jphi_Current_Cuda;
	Conformal_Current_Cuda* d_Jz_Current_Cuda;
	Conformal_Current_Cuda* d_Jr_Current_Cuda;
	Conformal_Current_Cuda* d_Jphi_Current_Cuda;
	vector<int> Z_Cuda;
	vector<int> R_Cuda;
	vector<int> Phi_Cuda;
	vector<int> offset;
	
	Array3D<T_Node_Cyl3D> node_info;
	T_Node_Cyl3D_Info* node_Conformal_info;
	// Array<T_Node_Cyl3D, 3> node_info;
	Array3D<TxVector<double> > E_node, B_node, E_node_pre, E_node_curr;
	// Array<TxVector<double>, 3> E_node, B_node, E_node_pre, E_node_curr;
	Array3D<double> Jz, Jr, Jphi, Rho;
	// Array<double, 3> Jz, Jr, Jphi, Rho;
	SI_SC_ComboEMFields_Cyl3D* em_field;
	Dynamic_ComboEMFieldsBase * EMFields;
	
	//tbb::concurrent_vector<tbb::concurrent_vector<tbb::concurrent_vector<double>>> Rho_t;
	
	Array3D<double> cell_volume, dual_cell_volume;
	// Array<double, 3> cell_volume, dual_cell_volume;
	
	Array2D<double> Bz_static, Br_static;
	// double **Bz_static, **Br_static;
	
	NodeField_Cyl3D(GridGeometry_Cyl3D * geom, SI_SC_ComboEMFields_Cyl3D* _em_field, ComboFieldsDefineRules * fldDefRules);
	~NodeField_Cyl3D();
	
	void setup_node();
	void BuildData();
	void CleanData();

	void BuildCUDADatas();
	void CleanCUDADatas();
	void fill_with_data();

	void Get_cuda_ptr(TxVector<double>** E_nodePtr, TxVector<double>** B_nodePtr){
        // *E_nodePtr  = dev_E_node_;
        // *B_nodePtr = dev_B_node_;
		*E_nodePtr  = d_E_node;
        *B_nodePtr = d_B_node;
    };

	float conformal_to_node_field_cuda();
	
	double get_cell_volume(int i_cell, int j_cell,int k_cell);
	double get_dual_cell_volume(int i_cell, int j_cell,int k_cell);
	
	void update(){
		conformal_to_node_field();
	}
	
	void conformal_to_node_field();
	
	void update_matrix()
	{
		conformal_to_node_field_matrix();
	}

	float update_cuda()
	{
		float time = conformal_to_node_field_cuda();
		return time;
	}

	void record_NodeField(std::ostream& out);

	void record_Current(std::ostream& out);

	void TransDgnData();

	void conformal_to_node_field_matrix();
	
	void step_to_conformal_current_test();
	
	void step_to_conformal_current_test_matrix();

	void step_to_conformal_current_test_cuda();

	void step_to_conformal_current_data();

	void fill_with_E(TxVector<double>& pos, TxVector<double>& e_field);
	void fill_with_E_CUDA(TxVector<double>& pos, TxVector<double>& e_field);
	void fill_with_B(TxVector<double>& pos, TxVector<double>& b_field);
	void fill_with_J(TxVector<double>& pos, TxVector<double>& density);
	void fill_with_J_CUDA(TxVector<double>& pos, TxVector<double>& density);
	void add_static_B(TxVector<double>& pos, TxVector<double>& b_field);
	
	void fill_with_E(const IndexAndWeights_Cyl3D& iw, TxVector<double>& e_field);
	void fill_with_E_CUDA(const IndexAndWeights_Cyl3D& iw, TxVector<double>& e_field);
	
	void fill_with_B(const IndexAndWeights_Cyl3D& iw, TxVector<double>& b_field);
	
	void fill_with_J(const IndexAndWeights_Cyl3D& iw, TxVector<double>& density);
	void fill_with_J_CUDA(const IndexAndWeights_Cyl3D& iw, TxVector<double>& density);
	
	void fill_with_Rho(TxVector<double>& pos, double& Rho);
	void fill_with_Rho_CUDA(TxVector<double>& pos, double& Rho);

	void clear_current_density();
	void clear_current_density_cuda();
	
	void accumulate_current_density(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& J_ptcl);
	
	void accumulate_I(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& disp_frac, double q2dt);
	void accumulate_I_CUDA(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& disp_frac, double q2dt);
	void accumulate_I(const vector<IndexAndWeights_Cyl3D>& iw_mid, const vector<TxVector<double> >& disp_frac,vector<double> q2dt);
	void accumulate_I_CUDA(const vector<IndexAndWeights_Cyl3D>& iw_mid, const vector<TxVector<double> >& disp_frac,vector<double> q2dt);
	void accumulate_Rho(const IndexAndWeights_Cyl3D& iw_mid, double q);


        //Set Static Bfield with Expression
	void setAttrib(const TxHierAttribSet& tas);

  	STFunc* m_Expressions[2];

	void setExpressionBNodeStatic();
	

void setBNodeStatic()
{
	double clsarr[6];
	double coil[6];
        coil[0] = -0.105;
        coil[1] = 0.295;
        coil[2] = 0.04;
        coil[3]	= 0.06;
        coil[4]	= 1000;
        coil[5]	= 2000;

	int form=1;

        clsarr[0] = coil[0];
        clsarr[1] = coil[1];
        clsarr[2] = coil[2];
        clsarr[3] = coil[3];
        clsarr[4] = coil[4];
        clsarr[5] = coil[5];

	switch(form)
	{
	case 1://left,right,bottom,top,Z-R
		clsarr[0] = 0.5*(coil[2]+coil[3]);
		clsarr[1] = 0.5*(coil[0]+coil[1]);
		clsarr[2] = coil[3]-coil[2];
		clsarr[3] = coil[1]-coil[0];
		break;
	case 2://bottom,top,left,right,R-Z
		clsarr[0] = 0.5*(coil[0]+coil[1]);
		clsarr[1] = coil[1]-coil[0];
		clsarr[2] = 0.5*(coil[2]+coil[3]);
		clsarr[3] = coil[3]-coil[2];
		break;
	case 3://z center,r center,z long, r long
		clsarr[0] = coil[1];
		clsarr[1] = coil[0];
		clsarr[2] = coil[3];
		clsarr[3] = coil[2];
		break;
	case 4://r center, z center, r long, z long
		default:
		break;
	}
	double* Bz = new double[(n_cell_z + 1) * (n_cell_r+1)];
       	double* Br = new double[(n_cell_z + 1) * (n_cell_r+1)];
       	for(int i=0;i<(n_cell_z + 1) * (n_cell_r+1);i++)
                Bz[i]=Br[i]=0;

	coils(clsarr,Bz,Br,n_cell_z,n_cell_r);
	for (int k = 0; k <= n_cell_r; k++)
       	{
       	for (int j = 0; j <= n_cell_z; j++)
       	{
       		Bz_static.SetValue(j, k, Bz[(n_cell_z+1) * k + j]);
			Br_static.SetValue(j, k, Br[(n_cell_z+1) * k + j]);
       	}
       }
      	delete Bz;
      	delete Br;

}


void coils(double *clsarr,double *Bz,double *Br,int J,int K)
{
	//only dealt with uniform girds
	double delt_z = global_grid->GetStep(0, 1);
	double delt_r = global_grid->GetStep(1, 1);

	int j1,j2;
	int kLayer,jTurn;
	// double r1,r2,delt_rc,a;
	double r1, delt_rc, a;
//********************************************************************************
	j1 = int((clsarr[1] - 0.5*clsarr[3])/delt_z);
	if(j1 <= 0.0)
		j1 = j1 - 1;
	j2 = int((clsarr[1] + 0.5*clsarr[3])/delt_z);
	if(j2 <= 0.0)
		j2 = j2 - 1;
	r1 = (clsarr[0] - 0.5*clsarr[2]);
	// r2 = (clsarr[0] + 0.5*clsarr[2]);

	kLayer = clsarr[2]/delt_r;

	if( kLayer > 1) 
	{
		if(kLayer > 10)
			kLayer = 10;
	}
	else
		kLayer = 1;
	delt_rc = clsarr[2]/kLayer;
	jTurn = j2-j1;
	if(jTurn == 0)
		jTurn = 1;
	a = 1e-7*clsarr[4]*clsarr[5]/(kLayer*jTurn);
//*******************************************************************************
	int j,k;
	int jT,kL;
	int jj,jk;
	double rc,r,dz;
	TxVector2D<double> Bzr;
	for(k = 0;k <= K;k++)
	{
		r = delt_r*k;
		rc = r1;
		for(kL = 0; kL < kLayer; kL++,rc += delt_rc)
		{
			for(j = -(j2-j1);j <= J;j++) 
			{
				dz = delt_z*(j-j1-1.5);

				Bzr=coil(r,rc,dz);

				for(jT = 0;jT < jTurn;jT++)
				{
					jj = j + jT;
					if(jj <= J && jj >= 0)
					{
						jk = (J+1)*k+jj;
						Bz[jk] = Bz[jk] + Bzr[0]*a;
						Br[jk] = Br[jk] + Bzr[1]*a;
					}
				}
			}
		}
	}
}

TxVector2D<double > coil(double r,double rl,double dz)
{
	double TWOPI=2.0*3.1415926;
	double bb=0.0,ab=1.0,rb=1.0,aari=1.0,ari=2.0,ak=2.0,d=0.0,s=0.0;
	double ge=0.0;
	TxVector2D<double> Bzr;
	if(r <= 0.0)//axis
	{
		s = (rl*rl + dz*dz);
		Bzr[0]= (TWOPI*rl*rl/(s*sqrt(s)));
		Bzr[1]= (0.0);
	}
	else
	{
		d = ((r-rl)*(r-rl) + dz*dz);
		s = r*rl*4.0 + d;
		ge = d/s;
		while((ge/aari) < 0.999999)
		{
			ak *= 2.0;
			ge = sqrt(ge*aari);
			ge *= 2.0;
			aari = ari;
			ari += ge;
			bb += rb*ge;
			bb *= 2.0;
			rb = ab;
			ab += bb / ari;
		}
		ge = TWOPI*rl/(sqrt(s)*ari);
		ak = (ak-ab)/s;
		ab /= d;
		Bzr[1] = (ab-ak)*ge*dz;
		Bzr[0] =((r+rl)*ak - (r-rl)*ab)*ge;
	}
	return Bzr;
}

	Array3D<TxVector<double> > Get_E_node(){
		return E_node;
	}

	Array3D<TxVector<double> > Get_B_node(){
		return B_node;
	}

	Array2D<double> Get_Bz_static(){
		return Bz_static;
	}

	Array2D<double> Get_Br_static(){
		return Br_static;
	}

	Array3D<double> Get_Jz(){
		return Jz;
	}

	Array3D<double> Get_Jr(){
		return Jr;
	}

	Array3D<double> Get_Jphi(){
		return Jphi;
	}

	Array3D<double> Get_Rho(){
		return Rho;
	}

	Array3D<double> Get_dual_cell_volume(){
		return dual_cell_volume;
	}
	
	void h_FillWithEfields(int ptclNum, 
						   IndexAndWeights_Cyl3D* ptclIndxWeights, 
						   TxVector<double> * efileds,
						   TxVector<double>* dev_E_node);

	void h_FillWithEfields(int ptclNum, cudaStream_t streamId, 
						   IndexAndWeights_Cyl3D* ptclIndxWeights, 
						   TxVector<double> * efileds,
                           TxVector<double>* dev_E_node);

	void h_FillWithBfields(int ptclNum, 
						   IndexAndWeights_Cyl3D* ptclIndxWeights, 
						   TxVector<double> * bfileds,
                           TxVector<double>* dev_B_node,
                           double* dev_Bz_static,
                           double* dev_Br_static);

	void h_FillWithBfields(int ptclNum, cudaStream_t streamId, 
						   IndexAndWeights_Cyl3D* ptclIndxWeights, 
						   TxVector<double> * bfileds,
                           TxVector<double>* dev_B_node,
                           double* dev_Bz_static,
                           double* dev_Br_static);

	void h_Species_Advance(int ptclNum, cudaStream_t streamId, 
						   IndexAndWeights_Cyl3D* ptclIndxWeights, 
						   TxVector<double>* dev_E_node, 
						   TxVector<double>* dev_B_node, 
						   double* dev_Bz_static, double* dev_Br_static,
						   TxVector<double>* dev_velocity, double dt);

	void h_Species_Accumulate(int ptclNum, cudaStream_t streamId, 
							  IndexAndWeights_Cyl3D* ptclIndxWeights, TxVector<double>* dev_position, 
							  TxVector<double>* dev_velocity, double* dev_weight, 
							  double dt, double q2dt, double* dev_Jz, double* dev_Jr, double* dev_Jphi, int* dev_cell_type, int* dev_rm_flag);

	void h_accelerate(int ptclNum, TxVector<double>* e_field, TxVector<double>* b_field, 
                                   TxVector<double>* dev_velocity, double dt, double chargeOverMass);

	void h_accelerate(int ptclNum, cudaStream_t streamId, TxVector<double>* e_field, TxVector<double>* b_field, 
                                   TxVector<double>* dev_velocity, double dt, double chargeOverMass);

	void h_translate_accumulate(int np, int* dev_idx_group, IndexAndWeights_Cyl3D* ptclIndxWeights, 
								TxVector<double>* dev_position, TxVector<double>* dev_velocity, double* dev_weight, 
								double dt, double q2dt, double* dev_Jz, double* dev_Jr, double* dev_Jphi, int* dev_cell_type, int* dev_rm_flag);

	void h_translate_accumulate(int np, IndexAndWeights_Cyl3D* ptclIndxWeights, 
								TxVector<double>* dev_position, TxVector<double>* dev_velocity, double* dev_weight, 
								double dt, double q2dt, double* dev_Jz, double* dev_Jr, double* dev_Jphi, int* dev_cell_type, int* dev_rm_flag);

	void h_translate_accumulate(int np, cudaStream_t streamId, IndexAndWeights_Cyl3D* ptclIndxWeights, 
								TxVector<double>* dev_position, TxVector<double>* dev_velocity, double* dev_weight, 
								double dt, double q2dt, double* dev_Jz, double* dev_Jr, double* dev_Jphi, int* dev_cell_type, int* dev_rm_flag);

	void h_Build_Group(int np, IndexAndWeights_Cyl3D* ptclIndxWeights, int* dev_idx_group0, int* dev_idx_group1, int* dev_idx_group2, 
                       int* dev_idx_group3, int* dev_idx_group4, int* dev_idx_group5, int* dev_idx_group6, int* dev_idx_group7, int* dev_flag);

	void h_test_accumulate(int np, double* dev_Jz, double* dev_Jr, double* dev_Jphi, int* dev_cell_type, int* dev_rm_flag);
};


#endif
