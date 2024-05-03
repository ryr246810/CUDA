#include <PtclBndInfo_Cyl3D.cuh>

#include <list>
// using namespace blitz;


PtclBndInfo_Cyl3D::PtclBndInfo_Cyl3D(GridGeometry_Cyl3D * geom){
	geometry = geom;
	global_grid = geom->GetZRGrid();
	n_cell_z = global_grid->GetDimension(0);
	n_cell_r = global_grid->GetDimension(1);
        m_phi_number= geometry->GetDimPhi();

	cell_type.InitArray(n_cell_z, n_cell_r, m_phi_number);
	bnd_edge.InitArray(n_cell_z, n_cell_r, m_phi_number);

	for(int i = 0; i < n_cell_z; ++i)
		for(int j = 0; j < n_cell_r; ++j)
			for(int k = 0; k < m_phi_number; ++k)
				cell_type.SetValue(i, j, k, 0);
	
	setup();
}

void PtclBndInfo_Cyl3D::setup(){
	list<EdgeData*> edge_list;
    for(int k=0;k<m_phi_number;k++){	
	int index = 0;
	for(int i = 0; i < n_cell_z; i++){
		for(int j = 0; j < n_cell_r; j++){
			vector<EdgeData*> *bnd_edge_tmp = bnd_edge.GetElemAddr(i, j, k);
			GridFace * face_ptr = geometry->GetGridGeometry(k)->GetGridFaces() + index;																
			const vector<GridFaceData*>& face_datas = face_ptr->GetFaces();
			int nf = face_datas.size();
			for(int iface = 0; iface < nf; iface++){
				if(face_datas[iface]->GetMaterialType() & PML || face_datas[iface]->GetMaterialType() & MUR){// ignore PML rgn appending edge data 
					continue;
				}
				
				const vector<AppendingEdgeData*>& apd_edge = face_datas[iface]->GetAppendingEdgeDatas();
				
				for(int ie = 0; ie < apd_edge.size(); ie++){
					bnd_edge_tmp->push_back(apd_edge[ie]);
					// bnd_edge(i, j, k).push_back(apd_edge[ie]);
					edge_list.push_back(apd_edge[ie]);
					// TxVector2D<double> pnt1 = apd_edge[ie]->GetFirstVertex()->GetLocation();
					// TxVector2D<double> pnt2 = apd_edge[ie]->GetLastVertex()->GetLocation();
					// cout<<"apd edge pnt : ("<<pnt1[0]<<" , "<<pnt1[1]<<") , ("<<pnt2[0]<<" , "<<pnt2[1]<<")"<<endl;
				}
				
				const vector<T_Element>& t_edges = face_datas[iface]->GetOutLineTEdge();
				for(int ie = 0; ie < t_edges.size(); ie++ ){
					GridEdgeData* the_edge = (GridEdgeData*)t_edges[ie].GetData();
					int material = the_edge->GetMaterialType();
					if(material & PML || material & MUR){
						// bnd_edge(i, j,k).push_back(the_edge);
						bnd_edge_tmp->push_back(the_edge);
						edge_list.push_back(the_edge);
					}
				}
			}
			

			if(bnd_edge_tmp->size()>0)
			{
				cell_type.SetValue(i, j, k, 1);
				// cell_type(i, j, k) = 1;
			}
			else if(face_datas.size() == 0)
			{
				cell_type.SetValue(i, j, k, 2);
				// cell_type(i, j, k) = 2;
			}
			if(((i-1)*(n_cell_z-1-1-i)<0)||(j>=n_cell_r-1))
			{
				cell_type.SetValue(i, j, k, 2);
				// cell_type(i, j, k) = 2;
			}
			
			index++;
			//cout<<cell_type(i, j, k);
		}
		//cout<<endl;
	}
	list<TxVector2D<double> > pnt_seq;
	
	list<EdgeData*>::iterator it = edge_list.begin();
	TxVector2D<double> pnt1 = (*it)->GetFirstVertex()->GetLocation();
	TxVector2D<double> pnt2 = (*it)->GetLastVertex()->GetLocation();
	pnt_seq.push_back(pnt1);
	pnt_seq.push_back(pnt2);
	edge_list.erase(it);
	
	int ne = edge_list.size();
	
	for(int i = 0; i < ne; i++){
		TxVector2D<double> start = *(pnt_seq.begin());
		TxVector2D<double> end = *(pnt_seq.rbegin());
		
		list<EdgeData*>::iterator it = edge_list.begin();
		while(it != edge_list.end() ){
			
			TxVector2D<double> p1 = (*it)->GetFirstVertex()->GetLocation();
			TxVector2D<double> p2 = (*it)->GetLastVertex()->GetLocation();
			if( (start - p2).length() < 1e-10 ){
				//cout<<"add pnt"<<endl;
				pnt_seq.push_front(p1);
				edge_list.erase(it);
				break;
			}
			else if( (end - p1).length() < 1e-10 ){
				//cout<<"add pnt"<<endl;
				pnt_seq.push_back(p2);
				edge_list.erase(it);
				break;
			}
			else if( (start - p1).length() < 1e-10 ){
				//cout<<"add pnt"<<endl;
				pnt_seq.push_front(p2);
				edge_list.erase(it);
				break;
			}
			else if( (end - p2).length() < 1e-10 ){
				//cout<<"add pnt"<<endl;
				pnt_seq.push_back(p1);
				edge_list.erase(it);
				break;
			}
			
			it++;
		}
	}
	/*
	stringstream ss;
	ss<<"Bnd_Cyl3D_pnt_seq"<<k<<".txt";
	fstream fout;
	fout.open(ss.str(), ios::out);
	int np = pnt_seq.size();
	//cout<<"pnt seq num = "<<np<<endl;
	list<TxVector2D<double> >::iterator itp = pnt_seq.begin();
	while(itp != pnt_seq.end() ){
		TxVector2D<double> tmp = *itp;
		fout<<tmp[0]<<" "<<tmp[1]<<endl;
		itp++;
		
	}
	fout.close();*/
}
}

bool PtclBndInfo_Cyl3D::check_sink(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell){	

	if(cell_type.GetValue(i_cell, j_cell, k_cell) == 0 ){
		return false;
	}
	else if(cell_type.GetValue(i_cell, j_cell, k_cell) == 2){
		return true;
	}
	else{
		vector<EdgeData*> *edge = bnd_edge.GetElemAddr(i_cell, j_cell, k_cell);
		int ne = edge->size();
		for(int i = 0; i < ne; i++){
			TxVector2D<double> pnt1 = (*edge)[i]->GetFirstVertex()->GetLocation();
			TxVector2D<double> pnt2 = (*edge)[i]->GetLastVertex()->GetLocation();
			TxVector2D<double> tmp_start= TxVector2D<double>(start_pos[0],start_pos[1]);
			TxVector2D<double> tmp_end= TxVector2D<double>(end_pos[0],end_pos[1]);
			if(detect_intersect_Cyl3D(pnt1, pnt2, tmp_start, tmp_end)){
				return true;
			}
		}
		return false;			
	}				
}

bool PtclBndInfo_Cyl3D::check_sink2(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell){	
	cout<<"--------in check sink 2------------------"<<endl;
	cout<<"cell type ("<<i_cell<<" , "<<j_cell<<") = "<<cell_type.GetValue(i_cell, j_cell,k_cell)<<endl;
	
	if(cell_type.GetValue(i_cell, j_cell, k_cell) == 0 ){
		cout<<"cell_type = "<<0<<endl;
		return false;
	}
	else if(cell_type.GetValue(i_cell, j_cell, k_cell) == 2){
		return true;
	}
	else{
		vector<EdgeData*> *edge = bnd_edge.GetElemAddr(i_cell, j_cell, k_cell);
		int ne = edge->size();
		
		cout<<"appending edge number = "<<ne<<endl;
		for(int i = 0; i < ne; i++){
			TxVector2D<double> pnt1 = (*edge)[i]->GetFirstVertex()->GetLocation();
			TxVector2D<double> pnt2 = (*edge)[i]->GetLastVertex()->GetLocation();

			TxVector2D<double> tmp_start= TxVector2D<double>(start_pos[0],start_pos[1]);
			TxVector2D<double> tmp_end= TxVector2D<double>(end_pos[0],end_pos[1]);
			
			cout<<"bnd edge pnt : ("<<pnt1[0]<<" , "<<pnt1[1]<<") , ("<<pnt2[0]<<" , "<<pnt2[1]<<")"<<endl; 
			
			if(detect_intersect_Cyl3D(pnt1, pnt2, tmp_start, tmp_end)){
				cout<<"---------------------interseted---------------------"<<endl;
				return true;
			}
		}
		cout<<"----------------not intersect-------------------"<<endl;
		return false;			
	}				
}

int PtclBndInfo_Cyl3D::check_sink3(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell){	
	if(cell_type.GetValue(i_cell, j_cell, k_cell) == 0 ){
		return 0;
	}
	else if(cell_type.GetValue(i_cell, j_cell, k_cell) == 2){
		return 2;
	}
	else{
		vector<EdgeData*> *edge = bnd_edge.GetElemAddr(i_cell, j_cell, k_cell);
		int ne = edge->size();
		for(int i = 0; i < ne; i++){
			TxVector2D<double> pnt1 = (*edge)[i]->GetFirstVertex()->GetLocation();
			TxVector2D<double> pnt2 = (*edge)[i]->GetLastVertex()->GetLocation();

			TxVector2D<double> tmp_start= TxVector2D<double>(start_pos[0],start_pos[1]);
			TxVector2D<double> tmp_end= TxVector2D<double>(end_pos[0],end_pos[1]);
			if(detect_intersect_Cyl3D(pnt1, pnt2, tmp_start, tmp_end)){
				return 1;
			}
		}
		return 0;			
	}				
}


