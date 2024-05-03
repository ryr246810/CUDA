#include <Species_Cyl3D.cuh>
#include <IndexAndWeights_Cyl3D.cuh>
#include <NodeField_Cyl3D.cuh>
#include <algorithm>
// #include "omp.h"

extern int ptclNum;

double gamma(const TxVector<double>& u){
return sqrt(1+u.Dot(u)*iSPEED_OF_LIGHT_SQ);// gamma is larger than 1
}

//int deleted_cnt;

Species_Cyl3D::Species_Cyl3D(NodeField_Cyl3D * field, ZRGrid * grid, GridGeometry_Cyl3D * geometry){
	node_field = field;
	global_grid = grid;
	geom = geometry;
	ptcl_bnd = new PtclBndInfo_Cyl3D(geom);
	
	ptgrp_capacity = 1024 * 1;
	PtclGroup_Cyl3D * pg = new PtclGroup_Cyl3D(ptgrp_capacity);//这样可以吗？
	ptcl_grps.push_back(pg);
	avail_grp = pg;
	
	m_efileds = new TxVector<double>[ptgrp_capacity];
	m_bfileds = new TxVector<double>[ptgrp_capacity];
	
	min_dist = 1e-10 * global_grid->GetMinStep();
	ptclNum = 0;
}
Species_Cyl3D::~Species_Cyl3D(){
	delete ptcl_bnd;
	delete[] m_efileds;
	delete[] m_bfileds;
	
	int n_grp = ptcl_grps.size();
	for(int i = 0; i < n_grp; i++){
		delete ptcl_grps[i];
	}
}
		
void Species_Cyl3D::setAttrib(const TxHierAttribSet& tha){
	vector< std::string > speciesNames = tha.getNamesOfType("Species");
	
	if(speciesNames.size() ){
		cout<<"Type of Species_Cyl3D is:   "<<speciesNames[0]<<endl;
		TxHierAttribSet attribs = tha.getAttrib(speciesNames[0]);
		if(attribs.hasString("name") ){
			name = attribs.getString("name");			
		}
		if(attribs.hasParam("charge") ){
			charge = attribs.getParam("charge");
		}
		if(attribs.hasParam("mass") ){
			mass = attribs.getParam("mass");			
		}
		if(attribs.hasParam("numPtclsInMacro") ){
			weight = attribs.getParam("numPtclsInMacro");
		}
		if(attribs.hasOption("mask") ){
			mask= attribs.getOption("mask");
		}
		else
		{
			mask=4;
		}
		if(attribs.hasParam("threshold") ){
			threshold= attribs.getParam("threshold");
		}
		else
		{
			threshold=5.0e6;
		}
	
		if(attribs.hasParam("edgeEnhance") ){
			edgeEnhance= attribs.getParam("edgeEnhance");
		}
		else
		{
			edgeEnhance=1.0;
		}
		chargeOverMass = charge / mass;
		macroCharge = weight * charge;
	}
}		


void Species_Cyl3D::advance(double dt){

	int n_grp = ptcl_grps.size();
	//#pragma omp parallel for
	for(int i = 0; i < n_grp; i++){
		// ptcl_grps[i]->resort_ptcl();
		fill_with_E(ptcl_grps[i]);
		fill_with_B(ptcl_grps[i]);
		accelerate(ptcl_grps[i], dt);
		// ptcl_grps[i]->resort_ptcl();
		translate_accumulate(ptcl_grps[i], dt);			
	}
	
	// n_grp = ptcl_grps.size();
	// // #pragma omp parallel for
	// for(int i = 0; i < n_grp; i++){
	// 	int np = ptcl_grps[i]->get_size();
	// 	ptcl_grps[i]->resort_ptcl();
	// 	for(int j = 0; j < np; j++){
	// 		node_field->accumulate_Rho(ptcl_grps[i]->idwt(j), macroCharge*ptcl_grps[i]->weight(j)); 
	// 	}
	// }
}

void Species_Cyl3D::record_PtclInfo(std::ostream& out)
{
	#ifdef __CUDA__
		TxVector<double> PtclInfo;
		TxVector<double> PtclPos;
		for(int i = 0; i < ptclNum; ++i){
			if(ptcl_cuda[0][i].rm_flag == 0){
				PtclInfo = ptcl_cuda[0][i].m_velocity;
				PtclPos = ptcl_cuda[0][i].m_position;

				out << PtclPos[0] << "\t\t" << PtclPos[1] << "\t\t" << PtclPos[2] << "\t\t" << PtclInfo[0] << "\t\t" << PtclInfo[1] << "\t\t" << PtclInfo[2] << "\n";
			}
		}
		out << "end!\n";
	#else
		TxVector<double> PtclInfo;
		TxVector<double> PtclPos;
		int n_grp = ptcl_grps.size();
		for(int i = 0; i < n_grp; ++i){
			int np = ptcl_grps[i]->get_size();
			for(int j = 0; j < np; ++j){
				PtclInfo = ptcl_grps[i]->velocity(j);
				PtclPos = ptcl_grps[i]->position(j);
				out << PtclPos[0] << "\t\t" << PtclPos[1] << "\t\t" << PtclPos[2] << "\t\t" << PtclInfo[0] << "\t\t" << PtclInfo[1] << "\t\t" << PtclInfo[2] << "\n";
			}
		}
		out << "end!\n";
	#endif	
}


void Species_Cyl3D::fill_with_E(PtclGroup_Cyl3D* ptcl_grp){
	int np = ptcl_grp->get_size();
	for(int i = 0; i < np; ++i){
		node_field->fill_with_E(ptcl_grp->idwt(i), m_efileds[i]);
	}
	
}


void Species_Cyl3D::fill_with_B(PtclGroup_Cyl3D* ptcl_grp){
	int np = ptcl_grp->get_size();
	for(int i = 0; i < np; ++i){
		node_field->fill_with_B(ptcl_grp->idwt(i), m_bfileds[i]);
		// node_field->add_static_B(ptcl_grp->position(i), m_bfileds[i]);
	}

}

void Species_Cyl3D::accelerate(PtclGroup_Cyl3D* ptcl_grp, double dt){		
	double f = 0.5 * dt * chargeOverMass;
	TxVector<double> u, u_prime;
	TxVector<double> a, t, s;

	int np = ptcl_grp->get_size();
	for(int i=0; i<np; i++){
		u = ptcl_grp->velocity(i);
		a  = m_efileds[i] * f;
		u += a;
		t = m_bfileds[i] * (f / gamma(u));//
		//t = m_bfileds[i] * f;//
		u_prime = u + u.Cross2(t);
		s = t * 2.0 / (1 + t.Dot(t) );
		u += u_prime.Cross2(s);
		u += a;
		ptcl_grp->velocity(i) = u;
	}
}

void Species_Cyl3D::translate_accumulate(PtclGroup_Cyl3D* ptcl_grp, double dt){
	double min_step = global_grid->GetMinStep();
	double q2dt = macroCharge / dt;
	TxVector<double> disp, vel,disp3 ;
	IndexAndWeights_Cyl3D iw_end;		
	TxVector<double> start_pos, end_pos, disp2;
	int n_segment;
	double fraction[4];
	IndexAndWeights_Cyl3D iw_segment[4];
			
	int np = ptcl_grp->get_size();				
	int flag;
	double r0, r1;
	double sin_alpha, cos_alpha;

	vector<int> rm_flag(np, 0);

	for(int i = 0; i < np; i += flag){
		flag = 1;
		start_pos = ptcl_grp->position(i);
		vel = ptcl_grp->velocity(i);
		disp = vel * (dt / gamma(vel));// used of accumulate current	
		r0 = start_pos[1] + disp[1];
		if(r0 < 0 ){
			ptcl_grp->velocity(i)[1] = -1.0 * vel[1];//modify velocity
			disp[1] = 0;
		}
		else{
			r1 = sqrt(r0 * r0 + disp[2] * disp[2]);
			if(r1 > 1.0e-22){
				sin_alpha = disp[2] / r1;
				cos_alpha = r0 / r1;
			}
			else{
				sin_alpha = 0;
				cos_alpha = 1;
			}
			// rotate velocity
			ptcl_grp->velocity(i)[1] =  cos_alpha * vel[1] + sin_alpha * vel[2];
			ptcl_grp->velocity(i)[2] = -sin_alpha * vel[1] + cos_alpha * vel[2];
		}			
		end_pos[0] = start_pos[0] + disp[0];		//z
		end_pos[1] = r1;							//r
		end_pos[2] = start_pos[2] + disp[2]/r1;		//phi(angle)

		disp2[0] = disp[0];			//变化量
		disp2[1] = r1-start_pos[1]; //变化量
		disp2[2] = disp[2]/r1;		//变化量

		disp3[0] = disp2[0];		//再次缓存变化量
		disp3[1] = disp2[1];
		disp3[2] = disp[2];
		
		if(disp.length() > min_step ){
			cout<<"error, disp.length > min_step "<<endl;
			getchar();
		}
		global_grid->ComputeIndexVecAndWeightsInGrid(end_pos, iw_end.indx, iw_end.wl, iw_end.wu );

		n_segment = frac_segment(ptcl_grp->idwt(i), iw_end, start_pos, disp2, fraction, iw_segment);
		
		// move	
		ptcl_grp->position(i) = end_pos;
		ptcl_grp->idwt(i) = iw_end;
		
		// accumulate
		int i_cell, j_cell, k_cell;
		int state;
		for(int i_seg = 0; i_seg < n_segment; i_seg++){
			i_cell = iw_segment[i_seg].indx[0];	//穿过的每个单元格都有三个方向
			j_cell = iw_segment[i_seg].indx[1];
			k_cell = iw_segment[i_seg].indx[2];
			
			TxVector<double> frac_disp = disp3 * fraction[i_seg];
			node_field->accumulate_I(iw_segment[i_seg], frac_disp , q2dt*ptcl_grp->weight(i));		
			
			if(ptcl_bnd->cell_type.GetValue(i_cell,j_cell,k_cell) == 2)
			{
				ptcl_grp->remove_ptcl(i);
				np--;
				flag = 0;// flag = 0  means ptcl be deleted
				break;
			}
		}	
	

		if(flag != 0){
			node_field->accumulate_Rho(ptcl_grp->idwt(i), macroCharge*ptcl_grp->weight(i)); 
		}
	
	}
}	


int Species_Cyl3D::frac_segment(IndexAndWeights_Cyl3D& idwt_start, 
				IndexAndWeights_Cyl3D& idwt_end,
				TxVector<double> & start_pos,
				TxVector<double> & disp, 
				double fraction[4], 
				IndexAndWeights_Cyl3D iw_seg[4])
{		
	int n_cross = 0;
	fraction[0] = fraction[1] = fraction[2] = fraction[3] = 1.0;
	
	for(int dir = 0; dir < 2; dir++){
		double step = global_grid->GetStep(dir, idwt_start.indx[dir]);// 这里有没有问题？？
		double distToSurf = (disp[dir] >= 0) ? (step * idwt_start.wl[dir]) : (step * idwt_start.wu[dir]);
		double dispA = fabs(disp[dir]);
		if(dispA > min_dist && dispA > distToSurf){
			fraction[n_cross] = distToSurf / dispA;
			n_cross++;			
		}
	}
    if(idwt_start.indx[2] != idwt_end.indx[2])
	{
		TxVector<double>end_pos = start_pos+disp;
        double factor_Phi;
 		global_grid->ComputeFactorCrossPhi(start_pos,idwt_start.indx[2],end_pos,idwt_end.indx[2], factor_Phi);
		double dispPhi = fabs(disp[2]);

		if(dispPhi> min_dist){
		  fraction[n_cross] = factor_Phi;
		  n_cross++;			
		}

    }	
	sort(fraction,fraction+n_cross); //排序
	for(int i = n_cross; i>0; i--){//nCross=0, 1, 2 3 三种情况都可以覆盖
		fraction[i]=fraction[i]-fraction[i-1];
	}
	int n_seg = n_cross + 1;
	
	TxVector<double> x0, x1, x_mid;
	x0 = start_pos;
	for(int i = 0; i < n_seg; i++){
		x1 = x0 + disp * fraction[i];
		x_mid = (x0 + x1) * 0.5;
		global_grid->ComputeIndexVecAndWeightsInGrid(x_mid, iw_seg[i].indx, iw_seg[i].wl, iw_seg[i].wu );
		x0 = x1;
	}
	return n_seg;
}

void Species_Cyl3D::test_grid(){
	int dim0 = global_grid->GetDimension(0);
	int dim1 = global_grid->GetDimension(1);
	int vdim0 = global_grid->GetVertexDimension(0);
	int vdim1 = global_grid->GetVertexDimension(1);
	int edim0 = global_grid->GetEdgeDimension(0, 0);
	int edim1 = global_grid->GetEdgeDimension(1, 1);
	cout<<"vdim0 "<<vdim0<<endl;
	cout<<"vdim1 "<<vdim1<<endl;
	
	TxVector2D<double> org = global_grid->GetOrg(); 
	cout<<"org coord 0 : "<<global_grid->GetOrg()[0]<<endl;
	cout<<"org coord 1 : "<<global_grid->GetOrg()[1]<<endl;
	
	fstream fout;
	fout.open("grid_location.txt", ios::out);
	
	fout<<"org 0 : "<<org[0]<<endl;
	fout<<"org 1 : "<<org[1]<<endl;
	
	fout<<"grid location 0 : "<<vdim0<<" vertex"<<endl;
	for(int i = 0; i < vdim0; i++){
		fout<<(global_grid->GetLength(0, i) + org[0])<<endl;
	}
	fout<<"grid location 1 : "<<vdim1<<" vertex"<<endl;
	for(int i = 0; i < vdim1; i++){
		fout<<(global_grid->GetLength(1, i) + org[1])<<endl;
	}
	fout.close();
	
	size_t theIndx;
	size_t theFrac;
	double the_dl;
	bool flag;
	
	TxVector<double> pos11(0, 0, 0.0);
	IndexAndWeights_Cyl3D iw11;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos11, iw11.indx, iw11.wl, iw11.wu);
	cout<<"pos11 loc"<<endl;
	cout<<iw11.indx[0]<<" "<<iw11.indx[1]<<" "<<iw11.wl[0]<<" "<<iw11.wl[1]<<endl;
}

void Species_Cyl3D::add_ptcl(double x[3], const IndexAndWeights_Cyl3D & idwt, double vel[3], int state){
	if(avail_grp->get_size() == ptgrp_capacity ){
		avail_grp = getAvailPG();
	}
	avail_grp->add_ptcl(x, idwt, vel, state);
}

void Species_Cyl3D::add_ptcl(const TxVector<double>& x, const IndexAndWeights_Cyl3D & idwt, const TxVector<double>& vel, int state){
	double pos[3] = {x[0], x[1], x[2]};
	double u[3] = {vel[0], vel[1], vel[2]};
	add_ptcl(pos, idwt, u, state);
}

void Species_Cyl3D::add_ptcl(double x[3], const IndexAndWeights_Cyl3D & idwt, double vel[3], int state,double wt){
	if(avail_grp->get_size() == ptgrp_capacity ){
		avail_grp = getAvailPG();
	}
	avail_grp->add_ptcl(x, idwt, vel, state,wt);
}

void Species_Cyl3D::add_ptcl(const TxVector<double>& x, const IndexAndWeights_Cyl3D & idwt, const TxVector<double>& vel, int state, double wt){
	double pos[3] = {x[0], x[1], x[2]};
	double u[3] = {vel[0], vel[1], vel[2]};
	add_ptcl(pos, idwt, u, state, wt);
}

PtclGroup_Cyl3D* Species_Cyl3D::getAvailPG(){
	int n_pg = ptcl_grps.size();
	for(int i = 0; i < n_pg; i++){
		if(ptcl_grps[i]->get_size() < ptgrp_capacity ){
			return ptcl_grps[i];
		}
	}
	PtclGroup_Cyl3D* new_pg = new PtclGroup_Cyl3D(ptgrp_capacity);
	ptcl_grps.push_back(new_pg);
	return new_pg;
}

void Species_Cyl3D::test_ptgrp(){
	double x[2] = {0, 0};
	IndexAndWeights_Cyl3D iw;
	double vel[3] = {0, 0, 0};
	for(int i = 0; i < 2048; i++){
		add_ptcl(x, iw, vel, 0);
	}
	
	ptcl_grps[0]->remove_ptcl(0);
	if(getAvailPG() ==  ptcl_grps[0]){
		cout<<"available pg is 0"<<endl;
	}
	int npg = ptcl_grps.size();
	cout<<npg<<" ptcl group"<<endl;
	for(int i = 0; i < npg; i++){
		cout<<"grp "<<i<<" size = "<<ptcl_grps[i]->get_size()<<endl;
	}	
}

void Species_Cyl3D::test_bnd(TxVector<double>& start_pos, TxVector<double>& end_pos){
	cout<<"-----------------in test bnd-------------------------- "<<endl;
	
	IndexAndWeights_Cyl3D iw_start, iw_end;
	TxVector<double> disp = end_pos - start_pos;
	global_grid->ComputeIndexVecAndWeightsInGrid(start_pos, iw_start.indx, iw_start.wl, iw_start.wu);
	global_grid->ComputeIndexVecAndWeightsInGrid(end_pos, iw_end.indx, iw_end.wl, iw_end.wu);
	double fraction[4];
	IndexAndWeights_Cyl3D iw_segment[4];
	
	
	int n_segment = frac_segment(iw_start, iw_end, start_pos, disp, fraction, iw_segment);
	cout<<"n_seg = "<<n_segment<<endl;
	
	TxVector<double> x0, x1, x_mid;
	x0 = start_pos;
	for(int i_seg = 0; i_seg < n_segment; i_seg++){
		x1 = x0 + disp * fraction[i_seg];
		
		x_mid = (x0 + x1) * 0.5;
		IndexAndWeights_Cyl3D iw_mid;
		global_grid->ComputeIndexVecAndWeightsInGrid(x_mid, iw_mid.indx, iw_mid.wl, iw_mid.wu);
		
		
		int i_cell = iw_segment[i_seg].indx[0];
		int j_cell = iw_segment[i_seg].indx[1];
		int k_cell = iw_segment[i_seg].indx[2];
		
		cout<<endl;
		cout<<"i_seg = "<<i_seg<<endl;
		cout<<" x0 = "<<x0[0]<<" "<<x0[1]<<endl;
		cout<<" x1 = "<<x1[0]<<" "<<x1[1]<<endl;
		cout<<"i_cell, j_cell = "<<i_cell<<" "<<j_cell<<"  , indx[0], indx[1] = "<<iw_mid.indx[0]<<" "<<iw_mid.indx[1]<<endl;
		
		int state = ptcl_bnd->check_sink2(start_pos, end_pos, i_cell, j_cell,k_cell);// meet the boundary?
		if(x1[0] > 0.1 &&  state == 0){
			cout<<"error, out of right bound, but not deleted"<<endl;
			cout<<"state = "<<state<<endl;
			cout<<"start pos = "<<start_pos[0]<<"  "<<start_pos[1]<<endl;
			cout<<"end   pos = "<<end_pos[0]<<"  "<<end_pos[1]<<endl;
			getchar();
		}
		x0 = x1;	
		cout<<endl;
	}
}

void Species_Cyl3D::test_move_accelerate(){
	cout<<"----------------in test move accelerate------------------------"<<endl;
	cout<<"org coord 0 : "<<global_grid->GetOrg()[0]<<endl;
	cout<<"org coord 1 : "<<global_grid->GetOrg()[1]<<endl;
	
	fstream fout;
	fout.open("test_move_accelerate.txt", ios::out);
	double dt = 1.0e-11;
	double x[2] = {0.05, 0.02};
	//double u[3] = {0, 1.0e8, 0};
	double u[3] = {0.6e8, -0.8e8, 0};
	
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(x, iw.indx, iw.wl, iw.wu);
	
	TxVector<double> pos;
	add_ptcl(x, iw ,u, 1);
	TxVector<double> disp;
	pos[0] = x[0];
	pos[1] = x[1];
	double dist_sum = 0;
	for(int i = 0; i < 100; i++){
		if(i % 5 == 0){
			cout<<i<<endl;
		}
		
		advance(dt);
		
		if(!count_ptcl()){
			cout<<"break advance"<<endl;
			break;
		}
		
		disp = ptcl_grps[0]->position(0) - pos;
		dist_sum += disp.length();
		pos = ptcl_grps[0]->position(0);
		fout<<pos[0]<<" "<<pos[1]<<endl;
	}
	cout<<"dist sum = "<<dist_sum<<endl;
	fout.close();
	cout<<"----------------end move accelerate------------------------"<<endl;
}

void Species_Cyl3D::test_advance(){
	cout<<"this is test advance"<<endl;
	TxVector<double> e_field(0, 0, 0);
	TxVector<double> b_field(2.0, 0, 0.0);
	
	TxVector2D<double> x(0.0, 1.5e-3);
	TxVector<double> u(3.0e8,0.0 , 0);
	
	TxVector<double> u_prime;
	TxVector<double> a, t, s;
	
	double dt = 0.414e-12;
	double f = 0.5 * dt * chargeOverMass;
	
	fstream fout;
	fout.open("test_advance.txt", ios::out);
	
	for(int i = 0; i < 1000; i++){
		if(i % 20 == 0){
			cout<<i<<endl;
		}
		a = e_field * f;
		u += a;
		t = b_field * (f / gamma(u));
		u_prime = u + u.Cross2(t);
		s = t * 2.0 / (t.Dot(t) + 1.0);
		u += u_prime.Cross2(s);
		u += a;
					
		TxVector<double> disp = u / gamma(u)  * dt;
		x[0] += disp[0];
		x[1] += disp[1];
					
		fout<<x[0]<<" "<<x[1]<<endl;				
	}		
	fout.close();

}
