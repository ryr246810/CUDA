#ifndef SPECIES_Cyl3D
#define SPECIES_Cyl3D

#include <TxVector2D.h>
#include <TxVector.h>
#include <TxStreams.h>

#include <TxHierAttribSet.h>

#include <PtclGroup_Cyl3D.cuh>
#include <IndexAndWeights_Cyl3D.cuh>

#include <ZRGrid.hxx>
#include <GridGeometry_Cyl3D.hxx>
#include <NodeField_Cyl3D.cuh>

#include <vector>
#include <string>

#include <PtclBndInfo_Cyl3D.cuh>
#include "Ptcl_CUDA_Cyl3D.cuh"

#define Ptcl_Grps_Num 1

class Species_Cyl3D
{
public:
	vector<Ptcl_CUDA_Cyl3D *> ptcl_cuda;
	int *h_d_rmQueue;
	int *d_rmNum;

	// physical level
	double mass;
	double charge;
	double chargeOverMass;
	string name;
	int mask;
	double threshold;
	double edgeEnhance;

	// macro ptcl level
	double weight;
	double macroCharge;
	int ptgrp_capacity;
	double min_dist;

	TxVector<double> *m_efileds;
	TxVector<double> *m_bfileds;

	vector<PtclGroup_Cyl3D *> ptcl_grps;
	PtclGroup_Cyl3D *avail_grp;

	NodeField_Cyl3D *node_field; // for fillWithE, fillWithB and accumulate current
	ZRGrid *global_grid;		 // for fill indexes and weights of ptcls
	// PtclGeomBndSinker* geomBndSink;
	PtclBndInfo_Cyl3D *ptcl_bnd;

	GridGeometry_Cyl3D *geom;

	// Species(NodeField * field, ZRGrid * grid, PtclGeomBndSinker* geom_sink){
	Species_Cyl3D(NodeField_Cyl3D *field, ZRGrid *grid, GridGeometry_Cyl3D *geometry);
	~Species_Cyl3D();

	void setAttrib(const TxHierAttribSet &tas);

	void advance(double dt);

	void record_PtclInfo(std::ostream &out);

	void fill_with_E(PtclGroup_Cyl3D *ptcl_grp);

	void fill_with_B(PtclGroup_Cyl3D *ptcl_grp);

	void accelerate(PtclGroup_Cyl3D *ptcl_grp, double dt);

	void translate_accumulate(PtclGroup_Cyl3D *ptcl_grp, double dt);

	int frac_segment(IndexAndWeights_Cyl3D &idwt_start,
					 IndexAndWeights_Cyl3D &idwt_end,
					 TxVector<double> &start_pos,
					 TxVector<double> &disp,
					 double fraction[4],
					 IndexAndWeights_Cyl3D idwts[4]);

	void test_grid();

	void add_ptcl(double x[3], const IndexAndWeights_Cyl3D &idwt, double vel[3], int state = 0);

	void add_ptcl(const TxVector<double> &x, const IndexAndWeights_Cyl3D &idwt, const TxVector<double> &vel, int state);

	void add_ptcl(double x[3], const IndexAndWeights_Cyl3D &idwt, double vel[3], int state, double wt);

	void add_ptcl(const TxVector<double> &x, const IndexAndWeights_Cyl3D &idwt, const TxVector<double> &vel, int state, double wt);

	void add_ptcl_CUDA(const TxVector<double> &x, const IndexAndWeights_Cyl3D &idwt, const TxVector<double> &vel, double wt);

	PtclGroup_Cyl3D *getAvailPG();

	void test_ptgrp();

	size_t count_ptcl_cuda();

	size_t count_ptcl()
	{
		size_t sum = 0;
		// #pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < ptcl_grps.size(); i++)
		{
			sum += ptcl_grps[i]->get_size();
		}
		return sum;
	}

	void out_ptcl_info(ostream &out, double time)
	{
		out << time << "\t";
		for (int ipg = 0; ipg < ptcl_grps.size(); ipg++)
		{
			int np = ptcl_grps[ipg]->get_size();
			for (int ip = 0; ip < np; ip++)
			{
				TxVector<double> pos = ptcl_grps[ipg]->position(ip);
				TxVector<double> vel = ptcl_grps[ipg]->velocity(ip);
				out << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\t" << vel[0] << "\t" << vel[0] << "\t" << endl;
			}
		}
	}

	void out_ptcl_info(string name, int n)
	{
		stringstream ss;
		ss << "ptcl_Cyl3D_pos_step_" << n << ".txt";
		name += "/" + ss.str();
		fstream fout;
		fout.open(name.c_str(), ios::out);
		for (int ipg = 0; ipg < ptcl_grps.size(); ipg++)
		{
			int np = ptcl_grps[ipg]->get_size();
			for (int ip = 0; ip < np; ip++)
			{
				TxVector<double> pos = ptcl_grps[ipg]->position(ip);
				TxVector<double> vel = ptcl_grps[ipg]->velocity(ip);
				fout << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\t" << vel[0] << "\t" << vel[1] << "\t" << vel[2] << "\t" << endl;
			}
		}
		fout.close();
	}

	void test_move_accelerate();

	void test_advance();

	void test_bnd(TxVector<double> &start_pos, TxVector<double> &end_pos);

	float Advance_With_Cuda(double dt);

	int Cuda_Init_Cuda_Constant_Vars();

	float advance_cuda_nonstream(double dt);

	float advance_cuda(double dt);

	void fill_with_data();

	void free_data();

	void Resize_PtclDatas(int n_grp);

	void fill_with_PtclDatas(int grpsNum, int maxPtclNum);

	void CopyPtclDatas_HostToDevice(int idx);

	void CopyPtclDatas_HostToDevice(cudaStream_t *streams, int idx);

	void CopyPtclDatas_DeviceToHost(int idx);
	void CopyPtclDatas_DeviceToHost(cudaStream_t *streams, int idx);

	void free_PtclDatas(int grpsNum);

	void CUDA_Push(int ptclNum, int grpIdx, double dt);

	void CUDA_Push_Ptcl(int ptclNum, cudaStream_t *streams, int grpIdx, double dt);

	void CUDA_Push(int ptclNum, cudaStream_t *streams, int grpIdx, double dt);

	void CUDA_Accumulate(PtclGroup_Cyl3D *ptcl_grp, int grpIdx, double dt);

	void CUDA_Accumulate(PtclGroup_Cyl3D *ptcl_grp, cudaStream_t *streams, int grpIdx, double dt);

	void CUDA_FillWithFields(int ptclNum, int grpIdx, double dt);

	void CUDA_FillWithFields(int ptclNum, cudaStream_t *streams, int grpIdx, double dt);

	void copy_to_host();

	void Translate_RemovePtcl(PtclGroup_Cyl3D *ptcl_grp);

	void BuildCUDADatas();

	void CleanCUDADatas();

protected:
	// CUDA Vars
	Array3D<TxVector<double>> *dev_E_fld;
	// TxVector<double>* dev_E_node;
	// TxVector<double>* dev_B_node;
	// double* dev_Bz_static;
	// double* dev_Br_static;
	// double* dev_Jz;
	// double* dev_Jr;
	// double* dev_Jphi;
	int *dev_cell_type;
	double *dev_dual_cell_volume;

	vector<IndexAndWeights_Cyl3D *> dev_ptclIndxWeights;

	vector<TxVector<double> *> dev_efileds;

	vector<TxVector<double> *> dev_bfileds;

	vector<TxVector<double> *> dev_position;

	vector<TxVector<double> *> dev_velocity;

	vector<double *> dev_weight;

	vector<int *> dev_rm_flag;
};

#endif
