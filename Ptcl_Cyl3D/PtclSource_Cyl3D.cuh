#ifndef PTCLSOURCE_Cyl3D
#define PTCLSOURCE_Cyl3D

#include <intersect_func_Cyl3D.cuh>
#include <ZRGrid.hxx>
#include <GridGeometry_Cyl3D.hxx>
#include <NodeField_Cyl3D.cuh>
#include <RandomFunc.hxx>
#include <AppendingEdgeData.hxx>
#include <BaseFunctionDefine.hxx>
#include <algorithm>


class Species_Cyl3D;

class EmitElement_Cyl3D{
public:
	EmitElement_Cyl3D(AppendingEdgeData* ape, const TxVector<double>& face_dir, const TxVector<double>& emitPnt,const double Phi){
		pnt1 = ape->GetFirstVertex()->GetLocation();
		pnt2 = ape->GetLastVertex()->GetLocation();
		emit_dir = face_dir;
		emit_pnt = emitPnt;
    	mid_pnt  = TxVector<double>(0.5*(pnt1[0]+pnt2[0]),0.5*(pnt1[1]+pnt2[1]),emit_pnt[2]);
		m_phi_number =int(0.5+4.0*asin(1.0)/Phi);
		emit_area = ape->GetSweptGeomDim()/m_phi_number;
		emit_flag = 0;
		residual_wt = 0.0;
		delt_phi = Phi;
	}

	TxVector2D<double> pnt1;
	TxVector2D<double> pnt2;
	
	TxVector<double> emit_pnt;
	TxVector<double> mid_pnt;
	TxVector<double> emit_dir;
	
	double delt_phi;
	double emit_area;
	int emit_flag;
	double residual_wt;
	double m_phi_number;
	bool edgeEnhance=false;
	
	void get_random_tan_dist(TxVector<double>& dis_t){
		double rand_num = randomFunc();
		TxVector2D<double> pos = pnt1 * rand_num + pnt2 * (1 - rand_num);
		double dis_0 = pos[0]- emit_pnt[0];
		double dis_1 = pos[1]- emit_pnt[1];

		TxVector<double> dis(dis_0,dis_1,0.0) ;
		dis_t = dis - emit_dir * (dis.Dot(emit_dir));
	}
	
	void get_random_emit_dist(TxVector<double>& dist){

		double rand_num = randomFunc();
		TxVector2D<double> pos_t = pnt1 * rand_num + pnt2 * (1 - rand_num);
		TxVector<double> pos(pos_t[0],pos_t[1],emit_pnt[2]);
                
		dist =emit_dir* (pos - emit_pnt).Dot(emit_dir) ;
	}
	
	void get_random_pos(TxVector<double>& pos){
		double rand_num = randomFunc();
		TxVector2D<double >pos_t=pnt1 * rand_num + pnt2 * (1 - rand_num);

		pos[0]=pos_t[0];
		pos[1]=pos_t[1];
		pos[2]=emit_pnt[2];
	}
	void get_init_emit_dist(TxVector<double>& dist){
		double rand_num = randomFunc();
		TxVector2D<double> pos_t = pnt1 * rand_num + pnt2 * (1 - rand_num);
		TxVector<double> pos (pos_t[0],pos_t[1],emit_pnt[2]);

		//dist = (pos - emit_pnt) * randomFunc();
		//dist = (pos - emit_pnt) ;
		dist =emit_dir*0.2* (pos - emit_pnt).Dot(emit_dir) ;
	}

	
};

class PtclSourceBase_Cyl3D{
public:
	GridGeometry_Cyl3D * geometry;
	const ZRGrid * global_grid;
	NodeField_Cyl3D * node_field;
	Species_Cyl3D * species;
	int emit_face_index;//由mask转换而来
	int m_phi_number;
	
	vector<EmitElement_Cyl3D*> emit_elements;
	
	PtclSourceBase_Cyl3D(GridGeometry_Cyl3D * geom, NodeField_Cyl3D * the_node_field, Species_Cyl3D * spe, int face_mask){
		geometry = geom;
		global_grid = geometry->GetZRGrid();
		node_field = the_node_field;
		species = spe;
		geometry->GetGridBndDatas()->ConvertFaceMasktoIndex(face_mask, emit_face_index);
		m_phi_number= geometry->GetDimPhi();
		setup_emitter();
		setup_edgehance();	
	}
	
	~PtclSourceBase_Cyl3D(){
		int n = emit_elements.size();
		for(int i = 0; i < n; i++){
			delete emit_elements[i];
		}
	}
	
	void setup_emitter(){
		 for(int k=0 ; k<m_phi_number; k++){
		 const GridGeometry *tmpgridGeom= geometry->GetGridGeometry(k);
		 GridFace* face_base = tmpgridGeom->GetGridFaces();
		 int n_cell_z = global_grid->GetDimension(0);
		 int n_cell_r = global_grid->GetDimension(1);
		 for(int i = 1; i < n_cell_z - 1; i++){
			 for(int j = 1; j < n_cell_r - 1; j++){
				 int index = i * n_cell_r + j;
				 GridFace* the_face = face_base + index;
				 if(!the_face->IsCut()){
					 continue;
				 }				 				 
				 const vector<GridFaceData*>& face_datas = the_face->GetFaces();
				 int nface = face_datas.size();
				 for(int iface = 0; iface < nface; iface++){
					 GridFaceData* the_data = face_datas[iface];
					 const vector<AppendingEdgeData*>& ape_datas = the_data->GetAppendingEdgeDatas();
					 for(int i_ape = 0; i_ape < ape_datas.size(); i_ape++){
						 if(ape_datas[i_ape]->HasFaceIndex(emit_face_index)){
							 double swept_vol = the_data->GetGeomDim() * the_data->GetDualGeomDim() ;
							 double cell_vol =   the_face->GetArea() * the_face->GetDualLength();
							 TxVector2D<double> normal_dir, emit_pnt;
							 TxVector<double> normal_dir_Cyl3D, emit_pnt_Cyl3D;
							 int flag = find_emit_info(the_face, ape_datas[i_ape], normal_dir, emit_pnt);
							 normal_dir_Cyl3D[0]=normal_dir[0];
							 normal_dir_Cyl3D[1]=normal_dir[1];
							 normal_dir_Cyl3D[2]=0.0;
							 double delt_Phi=4.0*asin(1.0)/m_phi_number;;
							 emit_pnt_Cyl3D[0]=emit_pnt[0];
							 emit_pnt_Cyl3D[1]=emit_pnt[1];
							 emit_pnt_Cyl3D[2]=delt_Phi*(0.5+k);
							 if(flag){
								 EmitElement_Cyl3D* tmp = new EmitElement_Cyl3D(ape_datas[i_ape], normal_dir_Cyl3D, emit_pnt_Cyl3D,delt_Phi);
								 emit_elements.push_back(tmp);
							 }
						 }
					 }
				 }
			 }
		 }		 
           }
	}


	
void ComputeEmissionElecField(EmitElement_Cyl3D* theEmitterElem, 
			 TxVector<double>& probePnt,
			 TxVector<double>& randomDis_L,
			 TxVector<double>& eField) const
{
  TxVector<double> tmpProbePnt;
  TxVector<double> tmpDL;
  TxVector<double> tmpEField;

  TxVector<double> emitDir = theEmitterElem->emit_dir;
  double emitDL   = global_grid->GetMinStep();

  eField = TxVector<double >(0.0, 0.0, 0.0) ;

  double tmpEmitEField = 0.;
  double emitEField=0.0;

  for (size_t j=0; j<10; ++j) {
    tmpDL = emitDir*emitDL*(0.1+0.9*randomFunc());
    tmpProbePnt = theEmitterElem->emit_pnt + tmpDL;

    
    node_field->fill_with_E(tmpProbePnt, tmpEField);
    tmpEmitEField = fabs(tmpEField.Dot(emitDir));
    if( tmpEmitEField>emitEField){
      probePnt = tmpProbePnt;
      randomDis_L = tmpDL;
      emitEField = tmpEmitEField;
      eField = tmpEField;
    }
  }
}

void ComputeEmissionElecField_CUDA(EmitElement_Cyl3D* theEmitterElem, 
			 TxVector<double>& probePnt,
			 TxVector<double>& randomDis_L,
			 TxVector<double>& eField) const
{
  TxVector<double> tmpProbePnt;
  TxVector<double> tmpDL;
  TxVector<double> tmpEField;

  TxVector<double> emitDir = theEmitterElem->emit_dir;
  double emitDL   = global_grid->GetMinStep();

  eField = TxVector<double >(0.0, 0.0, 0.0) ;

  double tmpEmitEField = 0.;
  double emitEField=0.0;

  for (size_t j=0; j<10; ++j) {
    tmpDL = emitDir*emitDL*(0.1+0.9*randomFunc());
    tmpProbePnt = theEmitterElem->emit_pnt + tmpDL;

    
    node_field->fill_with_E_CUDA(tmpProbePnt, tmpEField);
    tmpEmitEField = fabs(tmpEField.Dot(emitDir));
    if( tmpEmitEField>emitEField){
      probePnt = tmpProbePnt;
      randomDis_L = tmpDL;
      emitEField = tmpEmitEField;
      eField = tmpEField;
    }
  }
}


void ComputeEmissionElecField_gauss(EmitElement_Cyl3D* theEmitterElem, 
			 TxVector<double>& probePnt,
			 TxVector<double>& randomDis_L,
			 TxVector<double>& eField) const
{
  TxVector<double> tmpProbePnt;
  TxVector<double> tmpDL;
  TxVector<double> tmpEField;

  double tmpEmitEField = 0.;
  double emitEField=0.;

  TxVector<double> emitDir = theEmitterElem->emit_dir;
  TxVector<double> emit_pnt = theEmitterElem->emit_pnt;
  TxVector<double> mid_pnt = theEmitterElem->mid_pnt;
  double emitDL   = global_grid->GetMinStep();

  eField = TxVector<double>(0.0, 0.0, 0.0) ;

  for (size_t j=0; j<10; ++j) {
    tmpDL = emitDir*emitDL*randomFunc();
    //tmpProbePnt = theEmitterElem->emit_pnt + tmpDL;

    tmpProbePnt = mid_pnt + tmpDL;
    node_field->fill_with_E(tmpProbePnt, tmpEField);

    tmpEmitEField = fabs(tmpEField.Dot(emitDir));
    if( tmpEmitEField>emitEField){
      probePnt = tmpProbePnt;
      randomDis_L = tmpDL;
      emitEField = tmpEmitEField;
      eField = tmpEField;
    }
  }
}
	

	bool find_emit_info(GridFace* the_face, AppendingEdgeData* the_ape, TxVector2D<double>& normal_dir, TxVector2D<double>& emit_pnt){
		
		TxVector2D<double> pnt1 = the_ape->GetFirstVertex()->GetLocation();
		TxVector2D<double> pnt2 = the_ape->GetLastVertex()->GetLocation();
		TxVector2D<double> mid_pnt = (pnt1 + pnt2) * 0.5;
		TxVector2D<double> v12 = pnt2 - pnt1;
		double length = v12.length();
		double tol = global_grid->GetMinStep() * 1.0e-3;
		if(length < tol){
			return false;
		}
		v12 /= length;		
		normal_dir[0] = -1.0 * v12[1];// point to inner computation rgn
		normal_dir[1] = v12[0];		
		
		double diag_length = (the_face->GetVectorOfDir1() + the_face->GetVectorOfDir2()).length();
		TxVector2D<double> end_pnt = mid_pnt - normal_dir * diag_length; 
		
		GridEdge* edges[4];
		the_face->GetOutLineGridEdges(edges[0], edges[1], edges[2], edges[3]);//edge11, edge12, edge21, edge22
		for(int i = 0; i < 4; i++){
			GridVertexData * vertex[2];
			edges[i]->GetTwoEndGridVertices(vertex[0], vertex[1]);
			TxVector2D<double> edge_pnt0 = vertex[0]->GetLocation();
			TxVector2D<double> edge_pnt1 = vertex[1]->GetLocation();
			
			int is_intersected = detect_intersect_Cyl3D(mid_pnt, end_pnt, edge_pnt0, edge_pnt1);
			if(is_intersected){
				if(edges[i]->GetEdges().size() == 0){// full PEC edge 
					emit_pnt = get_intersect_pnt_Cyl3D(mid_pnt, end_pnt, edge_pnt0, edge_pnt1);
					//cout<<"emit pnt found, egde "<<i<<" "<<emit_pnt[0]<<" "<<emit_pnt[1]<<endl;
					return true;
				}
				else{// partial filled edge 
					if(vertex[0]->GetMaterialType() & PEC){
						emit_pnt = edge_pnt0;
						//cout<<"emit pnt found, egde "<<i<<" "<<emit_pnt[0]<<" "<<emit_pnt[1]<<endl;
						return true;
					}
					if(vertex[1]->GetMaterialType() & PEC){
						emit_pnt = edge_pnt1;
						//cout<<"emit pnt found, egde "<<i<<" "<<emit_pnt[0]<<" "<<emit_pnt[1]<<endl;
						return true;
					}
				}
				
			}
		}
		return false;	
	}
	
	
	virtual void emit(double dt){
		
	}
	void setup_edgehance(){
	vector<double> emit_pos;
	int n_elem = emit_elements.size();
	for(int i = 0; i < n_elem; i++){
	if(fabs(emit_elements[i]->emit_dir[0])>0.9)
	{
		emit_pos.push_back(emit_elements[i]->emit_pnt[1]);
	}
	else
	{
		emit_pos.push_back(emit_elements[i]->emit_pnt[0]);
	}

	}
	vector<double >::iterator lower_iter=min_element(emit_pos.begin(),emit_pos.end());	
	vector<double >::iterator upper_iter=max_element(emit_pos.begin(),emit_pos.end());	
	double tol = global_grid->GetMinStep() * 1.0e-3;
	for(int i = 0; i < n_elem; i++){
	if(fabs(emit_pos[i]-(*lower_iter))<tol ||fabs(emit_pos[i]-(*upper_iter))<tol){
	 emit_elements[i]->edgeEnhance=true;
	}
	}
	}

};

class BeamEmit_Cyl3D : public PtclSourceBase_Cyl3D{
public:
	double m_current_density;
	double m_voltage;
	double m_u;
	double m_gamma;
	
	BeamEmit_Cyl3D(GridGeometry_Cyl3D * geom, NodeField_Cyl3D * the_node_field, Species_Cyl3D * spe, int face_mask, double current_density, double voltage)
	: PtclSourceBase_Cyl3D(geom, the_node_field, spe, face_mask)
	{
		m_current_density = current_density;
		m_voltage = voltage;
		setup();
	}
	
	
	
	void setup();
	
	void emit(double dt);
};


class CLEmit_Cyl3D : public PtclSourceBase_Cyl3D{
public:
	double m_threshold;
	double m_edgehance;
	
	CLEmit_Cyl3D(GridGeometry_Cyl3D * geom, NodeField_Cyl3D * the_node_field, Species_Cyl3D * spe, int face_mask, double threshold,double edgehance)
	: PtclSourceBase_Cyl3D(geom, the_node_field, spe, face_mask)
	{
		m_threshold = threshold;
		m_edgehance =edgehance;
	}
	
	//void emit(double dt);
	//void emit_test(double dt);
	//void emit_wy(double dt);
	void emit_czg(double dt);
	void emit_cuda(double dt);
	void emit_gauss(double dt);
	void emit_gauss_cuda(double dt);
	void emit_gauss1(double dt);
	void emit_gauss_cuda1(double dt);
};




#endif


