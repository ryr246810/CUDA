#include <NodeField_Cyl3D.cuh>
#include <cuda_runtime.h>
#include<cmath>
using namespace std;
// #include "omp.h"
extern TxVector<double>* dev_E_node_;
extern TxVector<double>* dev_B_node_;
extern TxVector<double>* dev_E_node_pre_;
extern TxVector<double>* dev_E_node_curr_;
extern TxVector<double>* d_E_node;
extern TxVector<double>* d_B_node;
extern double* d_Jz;
extern double* d_Jr;
extern double* d_Jphi;
extern double* d_Rho;

NodeField_Cyl3D::NodeField_Cyl3D(GridGeometry_Cyl3D * geom, SI_SC_ComboEMFields_Cyl3D* _em_field, ComboFieldsDefineRules * fldDefRules){
	gridGeom_Cyl3D = geom;
	global_grid = geom->GetZRGrid();

	// for cuda
	m_Ezr_Ptr = NULL;
	m_Ephi_Ptr = NULL;
	m_Mzr_Ptr = NULL;
	m_Mphi_Ptr = NULL;

	m_h_d_node_Conformal_info = NULL;

	dev_E_node_ = NULL;
	dev_B_node_ = NULL;
	dev_E_node_pre_ = NULL;
	dev_E_node_curr_ = NULL;

	edges_z_current.clear();
	Jz_to_edges_z.clear();
	edges_r_current.clear();
	Jr_to_edges_r.clear();
	vertices_current.clear();
	Jphi_to_vertices.clear();

	em_field = _em_field;
	dynElecIndex = fldDefRules->Get_DynamicElecField_PhysDataIndex();
	dynMagIndex  = fldDefRules->Get_DynamicMagField_PhysDataIndex();
	currentIndex = fldDefRules->Get_J_PhysDataIndex();

	// printf("What happened?\n\n");
	EMFields = _em_field;
	EMFields->Get_Ezr_Info(&m_Ezr_Ptr, &m_Ezr_dataSize);
	EMFields->Get_Ephi_Info(&m_Ephi_Ptr, &m_Ephi_dataSize);
	EMFields->Get_Mzr_Info(&m_Mzr_Ptr, &m_Mzr_dataSize);
	EMFields->Get_Mphi_Info(&m_Mphi_Ptr, &m_Mphi_dataSize);

	// printf("What happened!!!\n\n");
	
	n_cell_z = global_grid->GetDimension(0);
	n_cell_r = global_grid->GetDimension(1);
    m_phi_number= gridGeom_Cyl3D->GetDimPhi();
	// printf("---%d  %d   %d \n", n_cell_z, n_cell_r, m_phi_number);
	// exit(-1);
	
	E_node.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	E_node_pre.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	E_node_curr.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	B_node.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	node_info.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);

	node_Conformal_info = new T_Node_Cyl3D_Info[(n_cell_z + 1) * (n_cell_r + 1) * m_phi_number];
	memset(node_Conformal_info, 0, sizeof(T_Node_Cyl3D_Info) * (n_cell_z + 1) * (n_cell_r + 1) * m_phi_number);

	h_Jz_Current_Cuda = new Conformal_Current_Cuda[(n_cell_z + 1) * (n_cell_r + 1) * m_phi_number];
	h_Jr_Current_Cuda = new Conformal_Current_Cuda[(n_cell_z + 1) * (n_cell_r + 1) * m_phi_number];
	h_Jphi_Current_Cuda = new Conformal_Current_Cuda[(n_cell_z + 1) * (n_cell_r + 1) * m_phi_number];

	for(int i = 0; i < (n_cell_z + 1) * (n_cell_r + 1) * m_phi_number; ++i){
		h_Jz_Current_Cuda[i].offset = -1;
		h_Jr_Current_Cuda[i].offset = -1;
		h_Jphi_Current_Cuda[i].offset = -1;
	}
	
	Jz.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	Jr.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	Jphi.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	Rho.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);
	
	for (int i = 0; i < (n_cell_z + 1); i++)
	{
		for (int j = 0; j < (n_cell_r + 1); j++)
		{
			for (int k = 0; k < m_phi_number; k++)
			{
				(*E_node.GetElemAddr(i, j, k)) = TxVector<double>(0, 0, 0);
				(*E_node_pre.GetElemAddr(i, j, k)) = TxVector<double>(0, 0, 0);
				(*E_node_curr.GetElemAddr(i, j, k)) = TxVector<double>(0, 0, 0);
			}
			
		}
	}
		
	clear_current_density();
	cell_volume.InitArray(n_cell_z, n_cell_r, m_phi_number);
	dual_cell_volume.InitArray(n_cell_z + 1, n_cell_r + 1, m_phi_number);

	for(int k =0; k < m_phi_number; k++){
		const GridGeometry *tmpGridGeom= gridGeom_Cyl3D->GetGridGeometry(k);
		int index = 0;
		for(int i = 0; i < n_cell_z; i++){
			for(int j = 0; j < n_cell_r; j++){
				GridFace* face_ptr = tmpGridGeom->GetGridFaces() + index;
				cell_volume.SetValue(i, j, k, face_ptr->GetArea() * face_ptr->GetDualLength() );
				index++;
			}
		}
	}
	
	for(int k =0; k < m_phi_number; k++)
	{
		const GridGeometry *tmpGridGeom= gridGeom_Cyl3D->GetGridGeometry(k);
		int index = 0;
		for(int i = 0; i < n_cell_z+1; i++){
			for(int j = 0; j < n_cell_r+1; j++){
				GridVertexData* vertex_ptr = tmpGridGeom->GetGridVertices() + index;
				Standard_Size currVertexIndex[2];
				vertex_ptr->GetVecIndex(currVertexIndex);
				double r_vertexLength = global_grid->GetCoordComp_From_VertexVectorIndx(1, currVertexIndex);
				if(currVertexIndex[1] == 1)
				{
					dual_cell_volume.SetValue(i, j, k, vertex_ptr->GetDualSweptGeomDim()*global_grid->GetStep(1,currVertexIndex[1])*mksConsts.pi/4.0 );
				}
				else if(currVertexIndex[1] == n_cell_r)
				{
					double dr=global_grid->GetStep(1,currVertexIndex[1]-1);
					dual_cell_volume.SetValue(i, j, k, vertex_ptr->GetDualSweptGeomDim()*mksConsts.pi*(r_vertexLength-dr/4.0)/m_phi_number );
				}
				else 
				{
					dual_cell_volume.SetValue(i, j, k, vertex_ptr->GetDualSweptGeomDim()*r_vertexLength*2.0*mksConsts.pi/m_phi_number );
				}
				index++;
			}
		}
	}

	setup_node();

    Bz_static.InitArray(n_cell_z + 1, n_cell_r + 1);
	Br_static.InitArray(n_cell_z + 1, n_cell_r + 1);

	Bz_static.SetAllValue(0.0);
	Br_static.SetAllValue(0.0);

}

NodeField_Cyl3D::~NodeField_Cyl3D()
{
	if(node_Conformal_info != NULL)
		delete[] node_Conformal_info;
	if (m_Ezr_Ptr != NULL) aligned_free(m_Ezr_Ptr);
	if (m_Ephi_Ptr != NULL) aligned_free(m_Ephi_Ptr);
	if (m_Mzr_Ptr != NULL) aligned_free(m_Mzr_Ptr);
	if (m_Mphi_Ptr != NULL) aligned_free(m_Mphi_Ptr);
	edges_z_current.clear();
	Jz_to_edges_z.clear();
	edges_r_current.clear();
	Jr_to_edges_r.clear();
	vertices_current.clear();
	Jphi_to_vertices.clear();
	// CleanData();
}

void NodeField_Cyl3D::setAttrib(const TxHierAttribSet& inputtha){
  std::vector< std::string > modelNames = inputtha.getNamesOfType("StaticNodeFLd");

  if( modelNames.size() ){
      std::cout << "\t Static NodeFld Setting Models are:";
      for(size_t i=0; i<modelNames.size(); ++i)
	std::cout << " " << modelNames[i];
      std::cout << std::endl;
    }
 for(size_t i=0; i<modelNames.size(); ++i){

    TxHierAttribSet tha= inputtha.getAttrib(modelNames[i]);

  std::vector< std::string > funcNames = tha.getNamesOfType("STFunc");

  for(Standard_Size j=0; j<funcNames.size(); j++){
    TxHierAttribSet attribs = tha.getAttrib(funcNames[j]);
    string functionName = attribs.getString("function");
    Standard_Integer index = -1;
    if(attribs.hasOption("component")){
      index  = attribs.getOption("component");
    }
    if( (index>3) || (index<0) ){
      continue;
    }
    try {
      m_Expressions[index] = TxMakerMap<STFunc>::getNew(functionName);
    }
    catch (TxDebugExcept& txde) {
      std::cout << txde << std::endl;
      return;
    }

    m_Expressions[index]->setAttrib(attribs);
  }


}  
  for(Standard_Size j=0; j<2; j++){
    if(m_Expressions[j]==NULL){
      m_Expressions[j] = new STFunc;
    }
  }
  setExpressionBNodeStatic();
}
void NodeField_Cyl3D::setExpressionBNodeStatic(){

	double x[2];
	for(Standard_Size i=0;i<n_cell_z+1;i++)
	for(Standard_Size j=0;j<n_cell_r+1;j++)
	{
		Standard_Size index[2]={i,j};
		x[0]=global_grid->GetCoordComp_From_VertexVectorIndx(0,index);//z
		x[1]=global_grid->GetCoordComp_From_VertexVectorIndx(1,index);//z
		Bz_static.SetValue(i, j, m_Expressions[0]->operator()(x, 0.0));
		Br_static.SetValue(i, j, m_Expressions[1]->operator()(x, 0.0));
	}
}

void NodeField_Cyl3D::conformal_to_node_field_matrix()
{
	
	int index;
	double value_e = 0.0, value_h = 0.0;
	for(int k = 0; k< m_phi_number; k++){
		for(int i = 1; i < n_cell_z; i++){
			for(int j = 1; j < n_cell_r; j++){
				TxVector<double> *the_E_node = E_node.GetElemAddr(i, j, k);
				TxVector<double> *the_E_node_pre = E_node_pre.GetElemAddr(i, j, k);
				TxVector<double> *the_E_node_curr = E_node_curr.GetElemAddr(i, j, k);
				TxVector<double> *the_B_node = B_node.GetElemAddr(i, j, k);

				// index = k * (n_cell_z + 1) * (n_cell_r + 1) + i * (n_cell_r + 1) + j;
				index = k * (n_cell_r + 1) * (n_cell_z + 1) + j * (n_cell_z + 1) + i;

				// Dir = 0, z direction
				value_e = 0.0;
				value_h = 0.0;
				value_e += (*node_Conformal_info[index].m_ElecEdge0[0]);
				value_e += (*node_Conformal_info[index].m_ElecEdge0[1]);

				value_h += (*node_Conformal_info[index].m_MagEdge0[0]);
				value_h += (*node_Conformal_info[index].m_MagEdge0[1]);
				value_h += (*node_Conformal_info[index].m_MagMinus0[0]);
				value_h += (*node_Conformal_info[index].m_MagMinus0[1]);

				(*the_E_node_curr)[0] = value_e * node_Conformal_info[index].tmpDataElec[0];
				(*the_B_node)[1] = value_h * node_Conformal_info[index].tmpDataMag1[0];

				// // Test
				// if(i == 10 && j == 10 && k == 1){
				// 	printf("E_node(10, 10, 1)[0] = %f\n", (*the_E_node_curr)[0]);
				// 	printf("B_node(10, 10, 1)[0] = %f\n", value_h * node_Conformal_info[index].tmpDataMag[0]);
				// }
				// // Test end

				// Dir = 1, r direction
				value_e = 0.0;
				value_h = 0.0;
				value_e += (*node_Conformal_info[index].m_ElecEdge1[0]);
				value_e += (*node_Conformal_info[index].m_ElecEdge1[1]);

				value_h += (*node_Conformal_info[index].m_MagEdge1[0]);
				value_h += (*node_Conformal_info[index].m_MagEdge1[1]);
				value_h += (*node_Conformal_info[index].m_MagMinus1[0]);
				value_h += (*node_Conformal_info[index].m_MagMinus1[1]);

				(*the_E_node_curr)[1] = value_e * node_Conformal_info[index].tmpDataElec[1];
				(*the_B_node)[0] = value_h * node_Conformal_info[index].tmpDataMag1[1];

				// Dir = 2, phi direction
				(*the_E_node_curr)[2] = (*node_Conformal_info[index].m_ElecVertex[0] + *node_Conformal_info[index].m_ElecVertex[1]) / 2;
				value_h = 0.0;
				value_h += (*node_Conformal_info[index].m_MagFace[0]);
				value_h += (*node_Conformal_info[index].m_MagFace[1]);
				value_h += (*node_Conformal_info[index].m_MagFace[2]);
				value_h += (*node_Conformal_info[index].m_MagFace[3]);

				(*the_B_node)[2] = value_h * node_Conformal_info[index].tmpDataMag[2];
				(*the_E_node) = ((*the_E_node_curr) + (*the_E_node_pre)) * 0.5;
				(*the_E_node_pre) = (*the_E_node_curr);

				// Test
				// if(i == 12 && j == 12 && k == 1){
				// 	printf("E_node(10, 10, 1)[0] = %f\n", (*the_E_node)[0]);
				// 	printf("B_node(10, 10, 1)[0] = %f\n", (*the_B_node)[2]);
				// }
				// Test end
			}
		}
	}

	// conformal_to_node_field_cuda();
}


// 设置一维类数组 node_Conformal_info 初始数据
void NodeField_Cyl3D::BuildData()
{
	int index;
	for(int k = 0; k < m_phi_number; ++k){
		const GridGeometry *currentGridGeom = gridGeom_Cyl3D->GetGridGeometry(k);
		const GridGeometry *minusGridGeom= currentGridGeom->GetMinusGridGeometry();
		for(int i = 1; i < n_cell_z; ++i){
			for(int j = 1; j < n_cell_r; ++j){
				
				T_Node_Cyl3D *node_info_tmp = node_info.GetElemAddr(i, j, k);
				GridVertexData* vertex_0 = currentGridGeom->GetGridVertices() + i * (n_cell_r + 1) + j;
				GridVertexData* vertex_1=  minusGridGeom->GetGridVertices() + i * (n_cell_r + 1) + j;
				node_info_tmp->vertex_data[0] = vertex_0 ;
				node_info_tmp->vertex_data[1] = vertex_1 ;

				index = k * (n_cell_r + 1) * (n_cell_z + 1) + j * (n_cell_z + 1) + i;
				// T_Node_Cyl3D *node_info_tmp = node_info.GetElemAddr(i, j, k);
				// GridVertexData *vertex_0 = (GridVertexData *)node_info_tmp->vertex_data[0];
				// GridVertexData *vertex_1 = (GridVertexData *)node_info_tmp->vertex_data[1];

				// Dir = 0, z director
				const vector<T_Element> &edge_0 = vertex_0->GetSharingDivTEdges(0); // 转存 emf 里面的数据， T_Element 专门存这种数据类型
				const vector<T_Element> &edge_1 = vertex_1->GetSharingDivTEdges(0);
				int ne0 = edge_0.size();
				int ne1 = edge_1.size();
				if(ne0 != 0){
					if(ne0 > 2){
						cout << "error in NodeField : set node - edge" << endl;
						getchar();
					}
					node_Conformal_info[index].num[0] = ne0;
					node_Conformal_info[index].tmpDataElec[0] = 1.0 / ne0;
					node_Conformal_info[index].tmpDataMag[0] = mksConsts.mu0 / ne0;

					for(int ie = 0; ie < ne0; ++ie){
						node_Conformal_info[index].m_ElecEdge0[ie] = edge_0[ie].GetData()->GetPhysDataPtr(dynElecIndex);
						node_Conformal_info[index].m_ElecEdge0_Ptr_Offset[ie] = node_Conformal_info[index].m_ElecEdge0[ie] - m_Ezr_Ptr;
						node_Conformal_info[index].m_MagEdge0[ie]  = edge_0[ie].GetData()->GetSweptPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagEdge0_Ptr_Offset[ie] = node_Conformal_info[index].m_MagEdge0[ie] - m_Mzr_Ptr;
					}
				}
				else{
					node_Conformal_info[index].tmpDataElec[0] = 0;
					node_Conformal_info[index].tmpDataMag[0] = 0;
				}

				if(ne1 != 0){
					if(ne1 > 2){
						cout << "error in NodeField : set node - edge" << endl;
						getchar();
					}
					node_Conformal_info[index].num1[0] = ne1;
					node_Conformal_info[index].tmpDataElec1[0] = 1.0 / (ne0 + ne1);
					node_Conformal_info[index].tmpDataMag1[0] = mksConsts.mu0 / (ne0 + ne1);

					for(int ie = 0; ie < ne1; ++ie){
						node_Conformal_info[index].m_MagMinus0[ie] = edge_1[ie].GetData()->GetSweptPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagMinus0_Ptr_Offset[ie] = node_Conformal_info[index].m_MagMinus0[ie] - m_Mzr_Ptr;
					}
				}
				else{
					node_Conformal_info[index].tmpDataElec1[0] = 0;
					node_Conformal_info[index].tmpDataMag1[0] = 0;
				}

				// Dir = 1, r direction
				const vector<T_Element> &edge0 = vertex_0->GetSharingDivTEdges(1); // 转存 emf 里面的数据， T_Element 专门存这种数据类型
				const vector<T_Element> &edge1 = vertex_1->GetSharingDivTEdges(1);
				ne0 = edge0.size();
				ne1 = edge1.size();
				if(ne0 != 0){
					if(ne0 > 2){
						cout << "error in NodeField : set node - edge" << endl;
						getchar();
					}
					node_Conformal_info[index].num[1] = ne0;
					node_Conformal_info[index].tmpDataElec[1] = 1.0 / ne0;
					node_Conformal_info[index].tmpDataMag[1] = mksConsts.mu0 / ne0;

					for(int ie = 0; ie < ne0; ++ie){
						node_Conformal_info[index].m_ElecEdge1[ie] = edge0[ie].GetData()->GetPhysDataPtr(dynElecIndex);
						node_Conformal_info[index].m_ElecEdge1_Ptr_Offset[ie] = node_Conformal_info[index].m_ElecEdge1[ie] - m_Ezr_Ptr;
						node_Conformal_info[index].m_MagEdge1[ie] = edge0[ie].GetData()->GetSweptPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagEdge1_Ptr_Offset[ie] = node_Conformal_info[index].m_MagEdge1[ie] - m_Mzr_Ptr;
					}
				}
				else{
					node_Conformal_info[index].tmpDataElec[1] = 0;
					node_Conformal_info[index].tmpDataMag[1] = 0;
				}

				if(ne1 != 0){
					if(ne1 > 2){
						cout << "error in NodeField : set node - edge" << endl;
						getchar();
					}
					node_Conformal_info[index].num1[1] = ne1;
					node_Conformal_info[index].tmpDataElec1[1] = 1.0 / (ne0 + ne1);
					node_Conformal_info[index].tmpDataMag1[1] = mksConsts.mu0 / (ne0 + ne1);

					for(int ie = 0; ie < ne1; ++ie){
						node_Conformal_info[index].m_MagMinus1[ie] = edge1[ie].GetData()->GetSweptPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagMinus1_Ptr_Offset[ie] = node_Conformal_info[index].m_MagMinus1[ie] - m_Mzr_Ptr;
					}
				}
				else{
					node_Conformal_info[index].tmpDataElec1[1] = 0;
					node_Conformal_info[index].tmpDataMag1[1] = 0;
				}

				// Dir = 2, phi direction
				const vector<GridFaceData *> &face0 = vertex_0->GetSharingGridFaceDatas();
				const vector<GridFaceData *> &face1 = vertex_1->GetSharingGridFaceDatas();
				int nf0 = face0.size();
				int nf1 = face1.size();

				if(nf0 != 0){
					if(nf0 > 4){
						cout << "error in NodeField : set node - face" << endl;
						cout << "nf = " << nf0 << endl;
						getchar();
					}
					node_Conformal_info[index].num[2] = nf0;
					node_Conformal_info[index].tmpDataMag[2] = mksConsts.mu0 / nf0;
					node_Conformal_info[index].m_ElecVertex[0] = vertex_0->GetSweptPhysDataPtr(dynElecIndex);
					node_Conformal_info[index].m_ElecVertex_Ptr_Offset[0] = node_Conformal_info[index].m_ElecVertex[0] - m_Ephi_Ptr;

					for(int iface = 0; iface < nf0; ++iface){
						node_Conformal_info[index].m_MagFace[iface] = face0[iface]->GetPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagFace_Ptr_Offset[iface] = node_Conformal_info[index].m_MagFace[iface] - m_Mphi_Ptr;
					}
				}
				else{
					node_Conformal_info[index].tmpDataMag[2] = 0;
				}

				if(nf1 != 0){
					if(nf1 > 4){
						cout << "error in NodeField : set node - face" << endl;
						cout << "nf = " << nf1 << endl;
						getchar();
					}
					node_Conformal_info[index].num1[2] = nf1;
					node_Conformal_info[index].tmpDataMag1[2] = mksConsts.mu0 / (nf0 + nf1);
					node_Conformal_info[index].m_ElecVertex[1] = vertex_1->GetSweptPhysDataPtr(dynElecIndex);
					node_Conformal_info[index].m_ElecVertex_Ptr_Offset[1] = node_Conformal_info[index].m_ElecVertex[1] - m_Ephi_Ptr;

					for(int iface = 0; iface < nf1; ++iface){
						node_Conformal_info[index].m_MagFace1[iface] = face1[iface]->GetPhysDataPtr(dynMagIndex);
						node_Conformal_info[index].m_MagFace1_Ptr_Offset[iface] = node_Conformal_info[index].m_MagFace1[iface] - m_Mphi_Ptr;
					}
				}

				node_Conformal_info[index].CheckDatas();
			}
		}
	}
}

void NodeField_Cyl3D::CleanData()
{
	int index;
	for(int k = 0; k < m_phi_number; ++k){
		const GridGeometry *currentGridGeom = gridGeom_Cyl3D->GetGridGeometry(k);
		const GridGeometry *minusGridGeom= currentGridGeom->GetMinusGridGeometry();
		for(int i = 1; i < n_cell_z; ++i){
			for(int j = 1; j < n_cell_r; ++j){

				index = k * (n_cell_z + 1) * (n_cell_r + 1) + i * (n_cell_r + 1) + j;
				T_Node_Cyl3D *node_info_tmp = node_info.GetElemAddr(i, j, k);
				GridVertexData *vertex_0 = (GridVertexData *)node_info_tmp->vertex_data[0];
				GridVertexData *vertex_1 = (GridVertexData *)node_info_tmp->vertex_data[1];

				// Dir = 0, z director
				const vector<T_Element> &edge_0 = vertex_0->GetSharingDivTEdges(0); // 转存 emf 里面的数据， T_Element 专门存这种数据类型
				const vector<T_Element> &edge_1 = vertex_1->GetSharingDivTEdges(0);
				int ne0 = edge_0.size();
				int ne1 = edge_1.size();
				if(ne0 != 0){
					for(int ie = 0; ie < ne0; ++ie){
						node_Conformal_info[index].m_ElecEdge0[ie] = NULL;
						node_Conformal_info[index].m_MagEdge0[ie] = NULL;
					}
				}
				if(ne1 != 0){
					for(int ie = 0; ie < ne1; ++ie){
						node_Conformal_info[index].m_MagMinus0[ie] = NULL;
					}
				}

				// Dir = 1, r direction
				const vector<T_Element> &edge0 = vertex_0->GetSharingDivTEdges(1); // 转存 emf 里面的数据， T_Element 专门存这种数据类型
				const vector<T_Element> &edge1 = vertex_1->GetSharingDivTEdges(1);
				ne0 = edge0.size();
				ne1 = edge1.size();
				if(ne0 != 0){
					for(int ie = 0; ie < ne0; ++ie){
						node_Conformal_info[index].m_ElecEdge1[ie] = NULL;
						node_Conformal_info[index].m_MagEdge1[ie] = NULL;
					}
				}
				if(ne1 != 0){
					for(int ie = 0; ie < ne1; ++ie){
						node_Conformal_info[index].m_MagMinus1[ie] = NULL;
					}
				}

				// Dir = 2, phi direction
				const vector<GridFaceData *> &face0 = vertex_0->GetSharingGridFaceDatas();
				const vector<GridFaceData *> &face1 = vertex_1->GetSharingGridFaceDatas();
				int nf0 = face0.size();
				int nf1 = face1.size();
				if(nf0 != 0){
					node_Conformal_info[index].m_ElecVertex[0] = NULL;
					for(int iface = 0; iface < nf0; ++iface){
						node_Conformal_info[index].m_MagFace[iface] = NULL;
					}
				}
				if(nf1 != 0){
					node_Conformal_info[index].m_ElecVertex[1] = NULL;
					for(int iface = 0; iface < nf1; ++iface){
						node_Conformal_info[index].m_MagFace1[iface] = NULL;
					}
				}
			}
		}
	}
}

void NodeField_Cyl3D::setup_node(){
	int index;
	int dir;
	for(int k=0; k< m_phi_number; k++){
		const GridGeometry *currentGridGeom= gridGeom_Cyl3D->GetGridGeometry(k);
		const GridGeometry *minusGridGeom= currentGridGeom->GetMinusGridGeometry();
		for(int i = 1; i < n_cell_z; i++){
			for(int j = 1; j < n_cell_r; j++){
				index = i * (n_cell_r + 1) + j;
				// index = k * (n_cell_z + 1) * (n_cell_r + 1) + i * (n_cell_r + 1) + j;
				T_Node_Cyl3D *node_info_tmp = node_info.GetElemAddr(i, j, k);
				GridVertexData* vertex_0 = currentGridGeom->GetGridVertices() + index;
				GridVertexData* vertex_1=  minusGridGeom->GetGridVertices() + index;
				node_info_tmp->vertex_data[0] = vertex_0 ;
				node_info_tmp->vertex_data[1] = vertex_1 ;
				// GridVertexData *vertex_0 = (GridVertexData *)node_info_tmp->vertex_data[0];
				// GridVertexData *vertex_1 = (GridVertexData *)node_info_tmp->vertex_data[1];
				for(dir = 0; dir < 2; dir++){
					const vector<T_Element>& edge_0 = vertex_0->GetSharingDivTEdges(dir);
					const vector<T_Element>& edge_1 = vertex_1->GetSharingDivTEdges(dir);
					int ne0 = edge_0.size();
					int ne1 = edge_1.size();
					if(ne0 != 0){
						if(ne0 > 2){
							cout<<"error in NodeField : set node - edge"<<endl;
							getchar();
						}
						node_info_tmp->flag = 1;
						for(int ie  = 0; ie < ne0; ie++){
							node_info_tmp->edge_data[dir][ie] = edge_0[ie].GetData();
						}
					}
					if(ne1 != 0){
						if(ne1 > 2){
							cout<<"error in NodeField : set node - edge"<<endl;
							getchar();
						}
						node_info_tmp->flag = 1;
						for(int ie  = 0; ie < ne1; ie++){
							node_info_tmp->minus_data[dir][ie] = edge_1[ie].GetData();
						}
					}
				}
				const vector<GridFaceData*>& face = vertex_0->GetSharingGridFaceDatas();
				int nf = face.size();
				
				if(nf != 0){
					if(nf > 4){
						cout<<"error in NodeField : set node - face"<<endl;
						cout<<"nf = "<<nf<<endl;
						getchar();
					}
					node_info_tmp->flag = 1;
					for(int iface = 0; iface < nf; iface++){
						node_info_tmp->face_data[iface] = face[iface];
					}
				}
			}
		}
   }
   BuildData();
   step_to_conformal_current_data();
}

double NodeField_Cyl3D::get_cell_volume(int i_cell, int j_cell, int k_cell){
	double vol_ijk = cell_volume.GetValue(i_cell, j_cell, k_cell);
	return vol_ijk;
}

double NodeField_Cyl3D::get_dual_cell_volume(int i_cell, int j_cell, int k_cell){
	double dual_vol_ijk = dual_cell_volume.GetValue(i_cell, j_cell, k_cell);
	return dual_vol_ijk;
}

void NodeField_Cyl3D::record_Current(std::ostream& out)
{
	#ifdef __CUDA__
		Standard_Real Jz_Tmp = d_Jz[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		Standard_Real Jr_Tmp = d_Jr[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		Standard_Real Jphi_Tmp = d_Jphi[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		Standard_Real Rho_Tmp = d_Rho[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		out << Jz_Tmp << "\t\t" << Jr_Tmp << "\t\t" << Jphi_Tmp << "\t\t" << Rho_Tmp << "\n";
	#else
		Standard_Real Jz_Tmp = Jz.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		Standard_Real Jr_Tmp = Jr.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		Standard_Real Jphi_Tmp = Jphi.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		Standard_Real Rho_Tmp = Rho.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		out << Jz_Tmp << "\t\t" << Jr_Tmp << "\t\t" << Jphi_Tmp << "\t\t" << Rho_Tmp << "\n";
	#endif
}

void NodeField_Cyl3D::record_NodeField(std::ostream& out)
{
	// out << "E_node \n";
	#ifdef __CUDA__
		TxVector<Standard_Real> E_nodeTmp = d_E_node[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		TxVector<Standard_Real> B_nodeTmp = d_B_node[(m_phi_number/2)*(n_cell_r+1)*(n_cell_z+1) + (n_cell_r/2)*(n_cell_z+1) + n_cell_z/2];
		out << E_nodeTmp[0] << "\t\t" << E_nodeTmp[1] << "\t\t" << E_nodeTmp[2] << "\t\t" << B_nodeTmp[0] << "\t\t" << B_nodeTmp[1] << "\t\t" << B_nodeTmp[2] << "\n";
	#else
		TxVector<Standard_Real> E_nodeTmp = E_node.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		TxVector<Standard_Real> B_nodeTmp = B_node.GetValue(n_cell_z/2, n_cell_r/2, m_phi_number/2);
		out << E_nodeTmp[0] << "\t\t" << E_nodeTmp[1] << "\t\t" << E_nodeTmp[2] << "\t\t" << B_nodeTmp[0] << "\t\t" << B_nodeTmp[1] << "\t\t" << B_nodeTmp[2] << "\n";
	#endif
}

void NodeField_Cyl3D::conformal_to_node_field(){
	int dir, ne, ie;//ne和ie暂时没有用
	double value_e=0.0, value_h=0.0;
	
	for(int k = 0; k< m_phi_number; k++){
	for(int i = 1; i < n_cell_z; i++){
	for(int j = 1; j < n_cell_r; j++){
			T_Node_Cyl3D the_node = node_info.GetValue(i, j, k);
			TxVector<double> *the_E_node = E_node.GetElemAddr(i, j, k);
			TxVector<double> *the_E_node_pre = E_node_pre.GetElemAddr(i, j, k);
			TxVector<double> *the_E_node_curr = E_node_curr.GetElemAddr(i, j, k);
			TxVector<double> *the_B_node = B_node.GetElemAddr(i, j, k);
			if(the_node.flag == 0){
				continue;
			}
			for(dir = 0; dir < 2; dir++){
				int dir2 = (dir + 1) % 2;
				value_e = 0;
				value_h = 0;
				int edge_esize=0;
				int edge_msize=0;
				if(the_node.edge_data[dir][0]){
					value_e += the_node.edge_data[dir][0]->GetPhysData(dynElecIndex);
					value_h += the_node.edge_data[dir][0]->GetSweptPhysData(dynMagIndex);
					edge_esize +=1;
				}
				
				if(the_node.edge_data[dir][1]){
					value_e += the_node.edge_data[dir][1]->GetPhysData(dynElecIndex);
					value_h += the_node.edge_data[dir][1]->GetSweptPhysData(dynMagIndex);
					edge_esize +=1;
				}
				edge_msize = edge_esize;
				if(the_node.minus_data[dir][0]){
					value_h += the_node.minus_data[dir][0]->GetSweptPhysData(dynMagIndex);
					edge_msize +=1;
				}
				if(the_node.minus_data[dir][1]){
					value_h += the_node.minus_data[dir][1]->GetSweptPhysData(dynMagIndex);
					edge_msize +=1;
				}
				
				if(edge_esize==0)
				{
					(*the_E_node_curr)[dir] = 0.0;				
				}
				else
				{
					(*the_E_node_curr)[dir] =  value_e/edge_esize;
				}

				if(edge_msize==0)
				{
					(*the_B_node)[dir2] = 0.0;
				}
				else
				{
					(*the_B_node)[dir2] =  value_h  * mksConsts.mu0/edge_msize;
				}
			}

			value_e = 0;
			int size_vertex= 0;

			for(int ivertex = 0; ivertex< 2; ivertex++){
				if(the_node.vertex_data[ivertex]){
					value_e += the_node.vertex_data[ivertex]->GetSweptPhysData(dynElecIndex);
					size_vertex +=1;
				}
			}

			if(size_vertex == 0)
			{
				(*the_E_node_curr)[2] = 0.0;
			}

			else
			{	
				(*the_E_node_curr)[2] =  value_e /size_vertex;
			}

			value_h = 0;
			int size_face = 0;
			for(int iface = 0; iface < 4; iface++){
				if(the_node.face_data[iface]){
					value_h += the_node.face_data[iface]->GetPhysData(dynMagIndex);
					size_face+=1;
				}
			}
			if(size_face==0)
			{
				(*the_B_node)[2] = 0.0;
			}

			else
			{	
				(*the_B_node)[2] =  value_h * mksConsts.mu0/size_face;
			}
			(*the_E_node) = ((*the_E_node_curr) + (*the_E_node_pre)) * 0.5;
			(*the_E_node_pre) = (*the_E_node_curr);
		}
	}
   }
}

void NodeField_Cyl3D::step_to_conformal_current_data(){
	vector<const GridGeometry*> tmpgridGeom;
	tmpgridGeom.clear();
	for(int k = 0; k < m_phi_number; ++k){
		tmpgridGeom.push_back(gridGeom_Cyl3D->GetGridGeometry(k));
	}
	int index, index0;

	for(int i = 1; i < n_cell_z - 1; ++i){
		for(int j = 1; j < n_cell_r - 1; ++j){
			int Jzindex = i * (n_cell_r + 1) + j;
			int Jrindex = Jzindex - i;	//i * n_cell_r + j
			int Jphiindex = Jzindex; 	//i * (n_cell_r+1) + j
			for(int k = 0; k < m_phi_number; ++k){
				index = k * (n_cell_r + 1) * (n_cell_z + 1) + j * (n_cell_z + 1) + i;
				// block 1
				{
					GridEdge * the_edge = tmpgridGeom[k]->GetGridEdges()[0] + Jzindex;
					const vector<GridEdgeData*>& data = the_edge->GetEdges();
					int n_edge = data.size();
					for(int i_edge = 0; i_edge < n_edge; ++i_edge){
						edges_z_current.push_back(data[i_edge]->GetPhysDataPtr(currentIndex));
						// edges_z_current[index] = data[i_edge]->GetPhysDataPtr(currentIndex);
						h_Jz_Current_Cuda[index].offset = data[i_edge]->GetPhysDataPtr(currentIndex) - m_Ezr_Ptr;
						Jz_to_edges_z.push_back(Jz.GetElemAddr(i, j, k));
						// Jz_to_edges_z[index] = Jz.GetElemAddr(i, j, k);
						h_Jz_Current_Cuda[index].z = i;
						h_Jz_Current_Cuda[index].r = j;
						h_Jz_Current_Cuda[index].phi = k;
					}
				}	
				// block 2
				{
					GridEdge * the_edge = tmpgridGeom[k]->GetGridEdges()[1] + Jrindex;
					const vector<GridEdgeData*>& data = the_edge->GetEdges();
					int n_edge = data.size();
					for(int i_edge = 0; i_edge < n_edge; ++i_edge){
						edges_r_current.push_back(data[i_edge]->GetPhysDataPtr(currentIndex));
						// edges_r_current[index] = data[i_edge]->GetPhysDataPtr(currentIndex);
						h_Jr_Current_Cuda[index].offset = data[i_edge]->GetPhysDataPtr(currentIndex) - m_Ezr_Ptr;
						Jr_to_edges_r.push_back(Jr.GetElemAddr(i, j, k));
						// Jr_to_edges_r[index] = Jr.GetElemAddr(i, j, k);
						h_Jr_Current_Cuda[index].z = i;
						h_Jr_Current_Cuda[index].r = j;
						h_Jr_Current_Cuda[index].phi = k;
					}
				}
				// block 3
				{
					GridVertexData *the_vertex = tmpgridGeom[k]->GetGridVertices() + Jphiindex;
					vertices_current.push_back(the_vertex->GetSweptPhysDataPtr(currentIndex));
					// vertices_current[index] = the_vertex->GetSweptPhysDataPtr(currentIndex);
					h_Jphi_Current_Cuda[index].offset = the_vertex->GetSweptPhysDataPtr(currentIndex) - m_Ephi_Ptr;
					Jphi_to_vertices.push_back(Jphi.GetElemAddr(i, j, k));
					// Jphi_to_vertices[index] = Jphi.GetElemAddr(i, j, k);
					h_Jphi_Current_Cuda[index].z = i;
					h_Jphi_Current_Cuda[index].r = j;
					h_Jphi_Current_Cuda[index].phi = k;
				}
			}
		}
	}
}

void NodeField_Cyl3D::step_to_conformal_current_test_matrix(){
	
	int size = edges_z_current.size();
	int size1 = (n_cell_z + 1) * (n_cell_r + 1) * m_phi_number;

	for(int i = 0; i < size1; ++i){
		if(h_Jz_Current_Cuda[i].offset != -1)
			m_Ezr_Ptr[h_Jz_Current_Cuda[i].offset] = *Jz.GetElemAddr(h_Jz_Current_Cuda[i].z, h_Jz_Current_Cuda[i].r, h_Jz_Current_Cuda[i].phi);
		if(h_Jr_Current_Cuda[i].offset != -1)
			m_Ezr_Ptr[h_Jr_Current_Cuda[i].offset] = *Jr.GetElemAddr(h_Jr_Current_Cuda[i].z, h_Jr_Current_Cuda[i].r, h_Jr_Current_Cuda[i].phi);
		if(h_Jphi_Current_Cuda[i].offset != -1)
			m_Ephi_Ptr[h_Jphi_Current_Cuda[i].offset] = *Jphi.GetElemAddr(h_Jphi_Current_Cuda[i].z, h_Jphi_Current_Cuda[i].r, h_Jphi_Current_Cuda[i].phi);
	}
	
	// for(int i = 0; i < size; ++i){
	// 	(*edges_z_current[i]) = (*Jz_to_edges_z[i]);
	// }

	// size = edges_r_current.size();
	// for(int i = 0; i < size; ++i){
	// 	(*edges_r_current[i]) = (*Jr_to_edges_r[i]);
	// }

	// size = vertices_current.size();
	// for(int i = 0; i < size; ++i){
	// 	(*vertices_current[i]) = (*Jphi_to_vertices[i]);
	// }
	
}

void NodeField_Cyl3D::step_to_conformal_current_test(){
	vector<const GridGeometry*> tmpgridGeom;
	tmpgridGeom.clear();
	for(int k=0; k<m_phi_number; k++)
	{
		tmpgridGeom.push_back(gridGeom_Cyl3D->GetGridGeometry(k));
	}
	// #pragma omp parallel for
	for(int i = 1; i < n_cell_z - 1; i++){
		for(int j = 1; j <= n_cell_r - 1; j++){
			int Jzindex = i * (n_cell_r + 1) + j;
			int Jrindex = Jzindex - i;	//i * n_cell_r + j
			int Jphiindex = Jzindex; 	//i * (n_cell_r+1) + j
			for(int k = 0; k<m_phi_number; k++){
				double the_Jz = Jz.GetValue(i, j, k);
				double the_Jr = Jr.GetValue(i, j, k);
				double the_Jphi = Jphi.GetValue(i, j, k);
				//block 1
				{
					GridEdge * the_edge = tmpgridGeom[k]->GetGridEdges()[0] + Jzindex;
					const vector<GridEdgeData*>& data = the_edge->GetEdges();
					int n_edge = data.size();
					for(int i_edge = 0; i_edge < n_edge; i_edge++){
						data[i_edge]->SetPhysData(currentIndex, the_Jz );
						//data[i_edge]->SetPhysData(currentIndex, 0.0 );
					}
				}
				//block 2
				{	
					GridEdge * the_edge = tmpgridGeom[k]->GetGridEdges()[1] + Jrindex;
					const vector<GridEdgeData*>& data = the_edge->GetEdges();
					int n_edge = data.size();
					for(int i_edge = 0; i_edge < n_edge; i_edge++){
						data[i_edge]->SetPhysData(currentIndex, the_Jr );
						//data[i_edge]->SetPhysData(currentIndex, 0.0 );
					}
				}
				//block 3
				{	
					GridVertexData * the_vertex= tmpgridGeom[k]->GetGridVertices() + Jphiindex;
					the_vertex->SetSweptPhysData(currentIndex, the_Jphi );
					//the_vertex->SetSweptPhysData(currentIndex, 0.0 );
				}		
			}
		}
   }
}

void NodeField_Cyl3D::fill_with_E(TxVector<double>& pos, TxVector<double>& e_field){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	fill_with_E(iw, e_field);
}	

void NodeField_Cyl3D::fill_with_E_CUDA(TxVector<double>& pos, TxVector<double>& e_field){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	fill_with_E_CUDA(iw, e_field);
}	

void NodeField_Cyl3D::add_static_B(TxVector<double>& pos, TxVector<double>& b_field){
	
	return;
	double x1=pos[0];
	double x2=pos[1];
        //b_field[0] += 2.8/(1+exp((x1-0.29)/0.02));
        //b_field[1] += 2.8/(2*0.02)*x2*exp((x1-0.29)/0.02)/powf(1+exp((x1-0.29)/0.02),2.0);

}			

void NodeField_Cyl3D::fill_with_B(TxVector<double>& pos, TxVector<double>& b_field){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	fill_with_B(iw, b_field);
}

void NodeField_Cyl3D::fill_with_J(TxVector<double>& pos, TxVector<double>& density){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	fill_with_J(iw, density);
}

void NodeField_Cyl3D::fill_with_J_CUDA(TxVector<double>& pos, TxVector<double>& density){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	fill_with_J_CUDA(iw, density);
}

void NodeField_Cyl3D::fill_with_E(const IndexAndWeights_Cyl3D& iw, TxVector<double>& e_field){
	
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	// printf("%d <-> %d\n", dr, rj);
	double r = rj + iw.wu[1]*dr;
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];
	
	if(j == 1){
		if(iw.wu[1] <= 0.5)
		{
			wu[0] = iw.wu[0];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
	}

	TxVector<double> E_node_ijk = E_node.GetValue(i, j, k);
	TxVector<double> E_node_i1jk = E_node.GetValue(i + 1, j, k);
	TxVector<double> E_node_ij1k = E_node.GetValue(i, j + 1, k);
	TxVector<double> E_node_i1j1k = E_node.GetValue(i + 1, j + 1, k);
	TxVector<double> E_node_ijk1 = E_node.GetValue(i, j, k1);
	TxVector<double> E_node_i1jk1 = E_node.GetValue(i + 1, j, k1);
	TxVector<double> E_node_ij1k1 = E_node.GetValue(i, j + 1, k1);
	TxVector<double> E_node_i1j1k1 = E_node.GetValue(i + 1, j + 1, k1);

	TxVector<double> e_temp = TxVector<double>(0.0,0.0,0.0);

	e_temp += E_node_ijk * (wl[0] * wl[1] * wl[2]) + E_node_i1jk * (wu[0] * wl[1] * wl[2]) \
	+ E_node_ij1k * (wl[0] * wu[1] * wl[2]) + E_node_i1j1k * (wu[0] * wu[1] * wl[2]);

	e_temp += E_node_ijk1 * (wl[0] * wl[1] * wu[2]) + E_node_i1jk1 * (wu[0] * wl[1] * wu[2]) \
	+ E_node_ij1k1 * (wl[0] * wu[1] * wu[2]) + E_node_i1j1k1 * (wu[0] * wu[1] * wu[2]);
	
	e_field = e_temp;
}			

void NodeField_Cyl3D::fill_with_E_CUDA(const IndexAndWeights_Cyl3D& iw, TxVector<double>& e_field){
	
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	// printf("%d <-> %d\n", dr, rj);
	double r = rj + iw.wu[1]*dr;
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];
	
	if(j == 1){
		if(iw.wu[1] <= 0.5)
		{
			wu[0] = iw.wu[0];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
	}
	
	TxVector<double> E_node_ijk = d_E_node[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];
	TxVector<double> E_node_i1jk = d_E_node[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1];
	TxVector<double> E_node_ij1k = d_E_node[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];
	TxVector<double> E_node_i1j1k = d_E_node[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1];
	TxVector<double> E_node_ijk1 = d_E_node[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];
	TxVector<double> E_node_i1jk1 = d_E_node[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1];
	TxVector<double> E_node_ij1k1 = d_E_node[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];
	TxVector<double> E_node_i1j1k1 = d_E_node[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1];

	// TxVector<double> E_node_ijk = E_node.GetValue(i, j, k);
	// TxVector<double> E_node_i1jk = E_node.GetValue(i + 1, j, k);
	// TxVector<double> E_node_ij1k = E_node.GetValue(i, j + 1, k);
	// TxVector<double> E_node_i1j1k = E_node.GetValue(i + 1, j + 1, k);
	// TxVector<double> E_node_ijk1 = E_node.GetValue(i, j, k1);
	// TxVector<double> E_node_i1jk1 = E_node.GetValue(i + 1, j, k1);
	// TxVector<double> E_node_ij1k1 = E_node.GetValue(i, j + 1, k1);
	// TxVector<double> E_node_i1j1k1 = E_node.GetValue(i + 1, j + 1, k1);

	TxVector<double> e_temp = TxVector<double>(0.0,0.0,0.0);

	e_temp += E_node_ijk * (wl[0] * wl[1] * wl[2]) + E_node_i1jk * (wu[0] * wl[1] * wl[2]) \
	+ E_node_ij1k * (wl[0] * wu[1] * wl[2]) + E_node_i1j1k * (wu[0] * wu[1] * wl[2]);

	e_temp += E_node_ijk1 * (wl[0] * wl[1] * wu[2]) + E_node_i1jk1 * (wu[0] * wl[1] * wu[2]) \
	+ E_node_ij1k1 * (wl[0] * wu[1] * wu[2]) + E_node_i1j1k1 * (wu[0] * wu[1] * wu[2]);
	
	e_field = e_temp;
}			

void NodeField_Cyl3D::fill_with_B(const IndexAndWeights_Cyl3D& iw, TxVector<double>& b_field){
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	
	Standard_Size zrIndex[2] = {Standard_Size(i),Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = rj + iw.wu[1]*dr;
	double wl[3], wu[3];

	wu[0] = iw.wu[0];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];

	if(j == 1)
	{
		if(iw.wu[1] <= 0.5)
		{
			wu[0] = iw.wu[0];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
	}

	TxVector<double> B_node_ijk = B_node.GetValue(i, j, k);
	TxVector<double> B_node_i1jk = B_node.GetValue(i + 1, j, k);
	TxVector<double> B_node_ij1k = B_node.GetValue(i, j + 1, k);
	TxVector<double> B_node_i1j1k = B_node.GetValue(i + 1, j + 1, k);
	TxVector<double> B_node_ijk1 = B_node.GetValue(i, j, k1);
	TxVector<double> B_node_i1jk1 = B_node.GetValue(i + 1, j, k1);
	TxVector<double> B_node_ij1k1 = B_node.GetValue(i, j + 1, k1);
	TxVector<double> B_node_i1j1k1 = B_node.GetValue(i + 1, j + 1, k1);

	TxVector<double> b_temp = TxVector<double>(0.0,0.0,0.0);

	b_temp += B_node_ijk * (wl[0] * wl[1] * wl[2]) + B_node_i1jk * (wu[0] * wl[1] * wl[2]) \
	+ B_node_ij1k * (wl[0] * wu[1] * wl[2]) + B_node_i1j1k * (wu[0] * wu[1] * wl[2]);

	b_temp += B_node_ijk1 * (wl[0] * wl[1] * wu[2]) + B_node_i1jk1 * (wu[0] * wl[1] * wu[2]) \
	+ B_node_ij1k1 * (wl[0] * wu[1] * wu[2]) + B_node_i1j1k1 * (wu[0] * wu[1] * wu[2]);
		
	b_temp[0] += Bz_static.GetValue(i, j) * (wl[0] * wl[1]) + Bz_static.GetValue(i + 1, j) * (wu[0] * wl[1]) \
	+ Bz_static.GetValue(i, j + 1) * (wl[0] * wu[1]) + Bz_static.GetValue(i + 1, j + 1) * (wu[0] * wu[1]);

	b_temp[1] += Br_static.GetValue(i, j) * (wl[0] * wl[1]) + Br_static.GetValue(i + 1, j) * (wu[0] * wl[1]) \
	+ Br_static.GetValue(i, j + 1) * (wl[0] * wu[1]) + Br_static.GetValue(i + 1, j + 1) * (wu[0] * wu[1]);
	b_field = b_temp;

}

void NodeField_Cyl3D::fill_with_Rho(TxVector<double>& pos, double& Rho_field){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = pos[1];
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];
	double Rho_ijk = Rho.GetValue(i, j, k);
	double Rho_i1jk = Rho.GetValue(i + 1, j, k);
	double Rho_ij1k = Rho.GetValue(i, j + 1, k);
	double Rho_i1j1k = Rho.GetValue(i + 1, j + 1, k);
	double Rho_ijk1 = Rho.GetValue(i, j, k1);
	double Rho_i1jk1 = Rho.GetValue(i + 1, j, k1);
	double Rho_ij1k1 = Rho.GetValue(i, j + 1, k1);
	double Rho_i1j1k1 = Rho.GetValue(i + 1, j + 1, k1);
	if(j == 1)
	{
		if(iw.wu[1] <= 0.5)
		{
			wu[0] = iw.wu[0];
			//wu[1] = iw.wu[1];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
		double Rho_tmp = 0.0;

		Rho_tmp +=  Rho_ijk * (wl[0] * wl[1] * wl[2]) + Rho_i1jk * (wu[0] * wl[1] * wl[2]) \
		+ Rho_ij1k * (wl[0] * wu[1] * wl[2]) + Rho_i1j1k * (wu[0] * wu[1] * wl[2]);				

		Rho_tmp += Rho_ijk1 * (wl[0] * wl[1] * wu[2]) + Rho_i1jk1 * (wu[0] * wl[1] * wu[2]) \
		+ Rho_ij1k1 * (wl[0] * wu[1] * wu[2]) + Rho_i1j1k1 * (wu[0] * wu[1] * wu[2]);	
		Rho_field = Rho_tmp;
	}
	else
	{	
		double Rho_tmp = 0.0;

		Rho_tmp +=  Rho_ijk * (wl[0] * wl[1] * wl[2]) + Rho_i1jk * (wu[0] * wl[1] * wl[2]) \
		+ Rho_ij1k * (wl[0] * wu[1] * wl[2]) + Rho_i1j1k * (wu[0] * wu[1] * wl[2]);				

		Rho_tmp += Rho_ijk1 * (wl[0] * wl[1] * wu[2]) + Rho_i1jk1 * (wu[0] * wl[1] * wu[2]) \
		+ Rho_ij1k1 * (wl[0] * wu[1] * wu[2]) + Rho_i1j1k1 * (wu[0] * wu[1] * wu[2]);	
		Rho_field = Rho_tmp;
		
	}
}

void NodeField_Cyl3D::fill_with_Rho_CUDA(TxVector<double>& pos, double& Rho_field){
	IndexAndWeights_Cyl3D iw;
	global_grid->ComputeIndexVecAndWeightsInGrid(pos, iw.indx, iw.wl, iw.wu);
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = pos[1];
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];

	double Rho_ijk = 	d_Rho[k *(n_cell_r+1)*(n_cell_z+1) + j    *(n_cell_z+1) + i];
	double Rho_i1jk = 	d_Rho[k *(n_cell_r+1)*(n_cell_z+1) + j    *(n_cell_z+1) + i+1];;
	double Rho_ij1k = 	d_Rho[k *(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];;
	double Rho_i1j1k = 	d_Rho[k *(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1];;
	double Rho_ijk1 = 	d_Rho[k1*(n_cell_r+1)*(n_cell_z+1) + j    *(n_cell_z+1) + i];;
	double Rho_i1jk1 = 	d_Rho[k1*(n_cell_r+1)*(n_cell_z+1) + j    *(n_cell_z+1) + i+1];;
	double Rho_ij1k1 = 	d_Rho[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];;
	double Rho_i1j1k1 = d_Rho[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1];;

	// double Rho_ijk = Rho.GetValue(i, j, k);
	// double Rho_i1jk = Rho.GetValue(i + 1, j, k);
	// double Rho_ij1k = Rho.GetValue(i, j + 1, k);
	// double Rho_i1j1k = Rho.GetValue(i + 1, j + 1, k);
	// double Rho_ijk1 = Rho.GetValue(i, j, k1);
	// double Rho_i1jk1 = Rho.GetValue(i + 1, j, k1);
	// double Rho_ij1k1 = Rho.GetValue(i, j + 1, k1);
	// double Rho_i1j1k1 = Rho.GetValue(i + 1, j + 1, k1);
	if(j == 1)
	{
		if(iw.wu[1] <= 0.5)
		{
			wu[0] = iw.wu[0];
			//wu[1] = iw.wu[1];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
		double Rho_tmp = 0.0;

		Rho_tmp +=  Rho_ijk * (wl[0] * wl[1] * wl[2]) + Rho_i1jk * (wu[0] * wl[1] * wl[2]) \
		+ Rho_ij1k * (wl[0] * wu[1] * wl[2]) + Rho_i1j1k * (wu[0] * wu[1] * wl[2]);				

		Rho_tmp += Rho_ijk1 * (wl[0] * wl[1] * wu[2]) + Rho_i1jk1 * (wu[0] * wl[1] * wu[2]) \
		+ Rho_ij1k1 * (wl[0] * wu[1] * wu[2]) + Rho_i1j1k1 * (wu[0] * wu[1] * wu[2]);	
		Rho_field = Rho_tmp;
	}
	else
	{	
		double Rho_tmp = 0.0;

		Rho_tmp +=  Rho_ijk * (wl[0] * wl[1] * wl[2]) + Rho_i1jk * (wu[0] * wl[1] * wl[2]) \
		+ Rho_ij1k * (wl[0] * wu[1] * wl[2]) + Rho_i1j1k * (wu[0] * wu[1] * wl[2]);				

		Rho_tmp += Rho_ijk1 * (wl[0] * wl[1] * wu[2]) + Rho_i1jk1 * (wu[0] * wl[1] * wu[2]) \
		+ Rho_ij1k1 * (wl[0] * wu[1] * wu[2]) + Rho_i1j1k1 * (wu[0] * wu[1] * wu[2]);	
		Rho_field = Rho_tmp;
		
	}
}


void NodeField_Cyl3D::fill_with_J(const IndexAndWeights_Cyl3D & iw, TxVector<double>& density){
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i),Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = rj + iw.wu[1]*dr;
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];
	double Jz_ijk = Jz.GetValue(i, j, k);//
	double Jz_ij1k = Jz.GetValue(i, j + 1, k);//
	double Jz_ijk1 = Jz.GetValue(i, j, k1);//
	double Jz_ij1k1 = Jz.GetValue(i, j + 1, k1);//

	double Jr_ijk = Jr.GetValue(i, j, k);//
	double Jr_i1jk = Jr.GetValue(i + 1, j, k);//
	double Jr_ijk1 = Jr.GetValue(i, j, k1);//
	double Jr_i1jk1 = Jr.GetValue(i + 1, j, k1);//

	double Jphi_ijk = Jphi.GetValue(i, j, k);//
	double Jphi_i1jk = Jphi.GetValue(i + 1, j, k);//
	double Jphi_ij1k = Jphi.GetValue(i, j + 1, k);//
	double Jphi_i1j1k = Jphi.GetValue(i + 1, j + 1, k);//
	if(j == 1)
	{
		if(iw.wu[1]<=0.5)
		{
			wu[0] = iw.wu[0];
			//wu[1] = iw.wu[1];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
		double density_tmp = 0.0;
	
		density[0] = Jz_ijk * wl[1]*wl[2] + Jz_ij1k *wu[1]*wl[2]+Jz_ijk1 * wl[1]*wu[2] + Jz_ij1k1 * wu[1]*wu[2];
		density[1] = Jr_ijk * wl[0]*wl[2] + Jr_i1jk *wu[0]*wl[2]+Jr_ijk1 * wl[0]*wu[2] + Jr_i1jk1 * wu[0]*wu[2];

		density_tmp += Jphi_ijk * (wl[0] * wl[1]) + Jphi_i1jk * (wu[0] * wl[1]) \
		+ Jphi_ij1k * (wl[0] * wu[1]) + Jphi_i1j1k * (wu[0] * wu[1]);				

		density[2]=density_tmp;
	}
	else
	{
		double density_tmp = 0.0;
	
		density[0] = Jz_ijk * wl[1]*wl[2] + Jz_ij1k *wu[1]*wl[2]+Jz_ijk1 * wl[1]*wu[2] + Jz_ij1k1 * wu[1]*wu[2];
		density[1] = Jr_ijk * wl[0]*wl[2] + Jr_i1jk *wu[0]*wl[2]+Jr_ijk1 * wl[0]*wu[2] + Jr_i1jk1 * wu[0]*wu[2];

		density_tmp += Jphi_ijk * (wl[0] * wl[1]) + Jphi_i1jk * (wu[0] * wl[1]) \
		+ Jphi_ij1k * (wl[0] * wu[1]) + Jphi_i1j1k * (wu[0] * wu[1]);	

		density[2]=density_tmp;
	}
}

void NodeField_Cyl3D::fill_with_J_CUDA(const IndexAndWeights_Cyl3D & iw, TxVector<double>& density){
	int i = iw.indx[0];
	int j = iw.indx[1];
	int k = iw.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i),Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = rj + iw.wu[1]*dr;
	double wl[3], wu[3];
	wu[0] = iw.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw.wu[2];
	wl[0] = iw.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw.wl[2];
	
	double Jz_ijk = d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];//
	double Jz_ij1k = d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];//
	double Jz_ijk1 = d_Jz[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];//
	double Jz_ij1k1 = d_Jz[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];//

	double Jr_ijk = d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];//
	double Jr_i1jk = d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1];//
	double Jr_ijk1 = d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];//
	double Jr_i1jk1 = d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1];//

	double Jphi_ijk = d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i];//
	double Jphi_i1jk = d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1];//
	double Jphi_ij1k = d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i];//
	double Jphi_i1j1k = d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1];//

	// double Jz_ijk = Jz.GetValue(i, j, k);//
	// double Jz_ij1k = Jz.GetValue(i, j + 1, k);//
	// double Jz_ijk1 = Jz.GetValue(i, j, k1);//
	// double Jz_ij1k1 = Jz.GetValue(i, j + 1, k1);//

	// double Jr_ijk = Jr.GetValue(i, j, k);//
	// double Jr_i1jk = Jr.GetValue(i + 1, j, k);//
	// double Jr_ijk1 = Jr.GetValue(i, j, k1);//
	// double Jr_i1jk1 = Jr.GetValue(i + 1, j, k1);//

	// double Jphi_ijk = Jphi.GetValue(i, j, k);//
	// double Jphi_i1jk = Jphi.GetValue(i + 1, j, k);//
	// double Jphi_ij1k = Jphi.GetValue(i, j + 1, k);//
	// double Jphi_i1j1k = Jphi.GetValue(i + 1, j + 1, k);//
	if(j == 1)
	{
		if(iw.wu[1]<=0.5)
		{
			wu[0] = iw.wu[0];
			//wu[1] = iw.wu[1];
			wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
			wu[2] = iw.wu[2];
			wl[0] = iw.wl[0];
			wl[1] = 1.0 - wu[1];
			wl[2] = iw.wl[2];
		}
		double density_tmp = 0.0;
	
		density[0] = Jz_ijk * wl[1]*wl[2] + Jz_ij1k *wu[1]*wl[2]+Jz_ijk1 * wl[1]*wu[2] + Jz_ij1k1 * wu[1]*wu[2];
		density[1] = Jr_ijk * wl[0]*wl[2] + Jr_i1jk *wu[0]*wl[2]+Jr_ijk1 * wl[0]*wu[2] + Jr_i1jk1 * wu[0]*wu[2];

		density_tmp += Jphi_ijk * (wl[0] * wl[1]) + Jphi_i1jk * (wu[0] * wl[1]) \
		+ Jphi_ij1k * (wl[0] * wu[1]) + Jphi_i1j1k * (wu[0] * wu[1]);				

		density[2]=density_tmp;
	}
	else
	{
		double density_tmp = 0.0;
	
		density[0] = Jz_ijk * wl[1]*wl[2] + Jz_ij1k *wu[1]*wl[2]+Jz_ijk1 * wl[1]*wu[2] + Jz_ij1k1 * wu[1]*wu[2];
		density[1] = Jr_ijk * wl[0]*wl[2] + Jr_i1jk *wu[0]*wl[2]+Jr_ijk1 * wl[0]*wu[2] + Jr_i1jk1 * wu[0]*wu[2];

		density_tmp += Jphi_ijk * (wl[0] * wl[1]) + Jphi_i1jk * (wu[0] * wl[1]) \
		+ Jphi_ij1k * (wl[0] * wu[1]) + Jphi_i1j1k * (wu[0] * wu[1]);	

		density[2]=density_tmp;
	}
}

void NodeField_Cyl3D::clear_current_density(){

	for(int i = 0; i < n_cell_z + 1; i++){
		for(int j = 0; j < n_cell_r + 1; j++){
			for(int k = 0; k < m_phi_number; k++){
				Jz.SetValue(i, j, k, 0.0);
				Jr.SetValue(i, j, k, 0.0);
				Jphi.SetValue(i, j, k, 0.0);
				Rho.SetValue(i, j, k, 0.0);
			}
		}
	}

	// for(int i = 0; i < n_cell_z + 1; i++){
	// 	for(int j = 0; j < n_cell_r + 1; j++){
	// 		for(int k = 0; k < m_phi_number; k++){
	// 			d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] = 0.0;
	// 			d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] = 0.0;
	// 			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] = 0.0;
	// 		}
	// 	}
	// }
	
}

void NodeField_Cyl3D::accumulate_current_density(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& J_ptcl){
	int i = iw_mid.indx[0];
	int j = iw_mid.indx[1];
	int k = iw_mid.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i),Standard_Size(j)};
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = rj + iw_mid.wu[1]*dr;
	double wl[3], wu[3];
	wu[0] = iw_mid.wu[0];
	//wu[1] = iw.wu[1];
	wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	wu[2] = iw_mid.wu[2];
	wl[0] = iw_mid.wl[0];
	wl[1] = 1.0 - wu[1];
	wl[2] = iw_mid.wl[2];
	
	Jz.AddValue(i, j, k, J_ptcl[0] * wl[1]*wl[2]);
	Jz.AddValue(i, j, k1, J_ptcl[0] * wl[1]*wu[2]);
	Jz.AddValue(i, j + 1, k, J_ptcl[0] * wu[1]*wl[2]);
	Jz.AddValue(i, j + 1, k1, J_ptcl[0] * wu[1]*wu[2]);

	Jr.AddValue(i, j, k, J_ptcl[1] * wl[0]*wl[2]); 		
	Jr.AddValue(i, j, k1, J_ptcl[1] * wl[0]*wu[2]); 		
	Jr.AddValue(i + 1, j, k, J_ptcl[1] * wu[0]*wl[2]);
	Jr.AddValue(i + 1, j, k1, J_ptcl[1] * wu[0]*wu[2]);
	
	Jphi.AddValue(i, j, k, J_ptcl[2] * (wl[0] * wl[1]*wl[2]) );
	Jphi.AddValue(i, j, k1, J_ptcl[2] * (wl[0] * wl[1]*wu[2]) );
	Jphi.AddValue(i + 1, j, k, J_ptcl[2] * (wu[0] * wl[1]*wl[2]) );
	Jphi.AddValue(i + 1, j, k1, J_ptcl[2] * (wu[0] * wl[1]*wu[2]) );
	Jphi.AddValue(i, j + 1, k, J_ptcl[2] * (wl[0] * wu[1]*wl[2]) );
	Jphi.AddValue(i, j + 1, k1, J_ptcl[2] * (wl[0] * wu[1]*wu[2]) );
	Jphi.AddValue(i + 1, j + 1, k, J_ptcl[2] * (wu[0] * wu[1]*wl[2]) );		
	Jphi.AddValue(i + 1, j + 1, k1, J_ptcl[2] * (wu[0] * wu[1]*wu[2]) );		

}	

void NodeField_Cyl3D::accumulate_I(const vector<IndexAndWeights_Cyl3D>& iw_mid, const vector<TxVector<double> >& disp_frac, vector<double> q2dt){


	int n_Segment= iw_mid.size();
	for(int i=0;i<n_Segment;i++){

		this->accumulate_I(iw_mid[i],disp_frac[i],q2dt[i]);	

	}	
}

void NodeField_Cyl3D::accumulate_I_CUDA(const vector<IndexAndWeights_Cyl3D>& iw_mid, const vector<TxVector<double> >& disp_frac, vector<double> q2dt){


	int n_Segment= iw_mid.size();
	for(int i=0;i<n_Segment;i++){

		this->accumulate_I_CUDA(iw_mid[i],disp_frac[i],q2dt[i]);	

	}	
}

void NodeField_Cyl3D::accumulate_I(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& disp_frac, double q2dt){
	int i = iw_mid.indx[0];
	int j = iw_mid.indx[1];
	int k = iw_mid.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	
	double dz = global_grid->GetStep(0, i);
	double dr = global_grid->GetStep(1, j);
	double dphi = 2.0*mksConsts.pi / m_phi_number;
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double rj1 = rj + dr;
	double r1 = rj + iw_mid.wu[1]*dr-disp_frac[1]/2.0;
	double r2 = rj + iw_mid.wu[1]*dr+disp_frac[1]/2.0;
	
	double del_z = disp_frac[0] / dz;	
	double del_phi = asin(disp_frac[2] / r2) / dphi;
	double mid_wu[3],mid_wl[3];
	double del_r, constnumber;
	{
		if(j == 1)
		{
			if(r1<=0.5*dr && r2<=0.5*dr)
			{
				del_r = 4.0*dr*dr*(r2-r1)*(r2+r1+dr)/((2*r1+dr)*(2*r1+dr)*(2*r2+dr)*(2*r2+dr));
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = 2*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr))+2*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr));
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r1>0.5*dr && r2>0.5*dr){
				del_r = (r2-r1)/(2.0*dr);
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = (r1+r2+2*dr)/(4.0*dr);
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r1<=0.5*dr && r2>=0.5*dr)
			{
				del_r = (r2+dr)/(2*dr)-4*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr));
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = (r2+dr)/(4*dr)+2*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr));
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r2<=0.5*dr && r1>=0.5*dr)
			{
				del_r = 4*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr))-(r1+dr)/(2*dr);
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = 2*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr))+(r1+dr)/(4*dr);
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}

			for(int m_phi=0; m_phi<m_phi_number; m_phi++)
			{
				Jz.AddValue(i, j, m_phi, del_z*(1-mid_wu[1])*q2dt);
			}
			Jz.AddValue(i, j + 1, k, (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt );
			Jz.AddValue(i, j + 1, k1, (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt );
			Jr.AddValue(i, j, k, (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt );
			Jr.AddValue(i, j, k1, (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt );
			Jr.AddValue(i + 1, j, k, (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt );
			Jr.AddValue(i + 1, j, k1, (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt );
			Jphi.AddValue(i, j, k, 0.0);
			Jphi.AddValue(i + 1, j, k, 0.0);
			Jphi.AddValue(i, j + 1, k, (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt );
			Jphi.AddValue(i + 1, j + 1, k, (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt );

		}
		else
		{
			del_r = (r1*r2+rj*rj1) * disp_frac[1] / (2*dr*r1*r2);
			constnumber = del_z*del_r*del_phi/12.0;
			mid_wu[0] = iw_mid.wu[0];
			mid_wu[1] = (r1-rj)*(rj1+r1)/(4*dr*r1)+(r2-rj)*(rj1+r2)/(4*dr*r2);
			mid_wu[2] = iw_mid.wu[2];
			mid_wl[0] = iw_mid.wl[0];
			mid_wl[1] = 1.0 - mid_wu[1];
			mid_wl[2] = iw_mid.wl[2];
			Jz.AddValue(i, j, k, (del_z*(1-mid_wu[1]-mid_wu[2]+mid_wu[1]*mid_wu[2])+constnumber)*q2dt );
			// printf("Jz(%d, %d, %d) = %f \n", i, j, k1, Jz.GetValue(i, j, k1));
			Jz.AddValue(i, j, k1, (del_z*(1-mid_wu[1])*mid_wu[2]-constnumber)*q2dt );
			Jz.AddValue(i, j + 1, k, (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt );
			Jz.AddValue(i, j + 1, k1, (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt );
			Jr.AddValue(i, j, k, (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt );
			Jr.AddValue(i, j, k1, (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt );
			Jr.AddValue(i + 1, j, k, (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt );
			Jr.AddValue(i + 1, j, k1, (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt );
			Jphi.AddValue(i, j, k, (del_phi*(1-mid_wu[0]-mid_wu[1]+mid_wu[0]*mid_wu[1])+constnumber)*q2dt );
			Jphi.AddValue(i + 1, j, k, (del_phi*mid_wu[0]*(1-mid_wu[1])-constnumber)*q2dt );
			Jphi.AddValue(i, j + 1, k, (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt );
			Jphi.AddValue(i + 1, j + 1, k, (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt );
		}
	}
	
}

void NodeField_Cyl3D::accumulate_I_CUDA(const IndexAndWeights_Cyl3D& iw_mid, const TxVector<double>& disp_frac, double q2dt){
	int i = iw_mid.indx[0];
	int j = iw_mid.indx[1];
	int k = iw_mid.indx[2];
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	
	double dz = global_grid->GetStep(0, i);
	double dr = global_grid->GetStep(1, j);
	double dphi = 2.0*mksConsts.pi / m_phi_number;
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double rj1 = rj + dr;
	double r1 = rj + iw_mid.wu[1]*dr-disp_frac[1]/2.0;
	double r2 = rj + iw_mid.wu[1]*dr+disp_frac[1]/2.0;
	
	double del_z = disp_frac[0] / dz;	
	double del_phi = asin(disp_frac[2] / r2) / dphi;
	double mid_wu[3],mid_wl[3];
	double del_r, constnumber;
	{
		if(j == 1)
		{
			if(r1<=0.5*dr && r2<=0.5*dr)
			{
				del_r = 4.0*dr*dr*(r2-r1)*(r2+r1+dr)/((2*r1+dr)*(2*r1+dr)*(2*r2+dr)*(2*r2+dr));
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = 2*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr))+2*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr));
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r1>0.5*dr && r2>0.5*dr){
				del_r = (r2-r1)/(2.0*dr);
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = (r1+r2+2*dr)/(4.0*dr);
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r1<=0.5*dr && r2>=0.5*dr)
			{
				del_r = (r2+dr)/(2*dr)-4*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr));
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = (r2+dr)/(4*dr)+2*r1*(r1+dr)/((2*r1+dr)*(2*r1+dr));
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			else if(r2<=0.5*dr && r1>=0.5*dr)
			{
				del_r = 4*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr))-(r1+dr)/(2*dr);
				constnumber = del_z*del_r*del_phi/12.0;
				mid_wu[0] = iw_mid.wu[0];
				mid_wu[1] = 2*r2*(r2+dr)/((2*r2+dr)*(2*r2+dr))+(r1+dr)/(4*dr);
				mid_wu[2] = iw_mid.wu[2];
				mid_wl[0] = iw_mid.wl[0];
				mid_wl[1] = 1.0 - mid_wu[1];
				mid_wl[2] = iw_mid.wl[2];
				
			}
			
			for(int m_phi=0; m_phi<m_phi_number; m_phi++)
			{
				d_Jz[m_phi*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += del_z*(1-mid_wu[1])*q2dt;
				// Jz.AddValue(i, j, m_phi, del_z*(1-mid_wu[1])*q2dt);
			}
			d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt;
			d_Jz[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt;
			d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt;
			d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt;
			d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt;
			d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += 0.0;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += 0.0;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1] += (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt;

			// Jz.AddValue(i, j + 1, k, (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt );
			// Jz.AddValue(i, j + 1, k1, (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt );
			// Jr.AddValue(i, j, k, (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt );
			// Jr.AddValue(i, j, k1, (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt );
			// Jr.AddValue(i + 1, j, k, (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt );
			// Jr.AddValue(i + 1, j, k1, (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt );
			// Jphi.AddValue(i, j, k, 0.0);
			// Jphi.AddValue(i + 1, j, k, 0.0);
			// Jphi.AddValue(i, j + 1, k, (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt );
			// Jphi.AddValue(i + 1, j + 1, k, (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt );

		}
		else
		{
			del_r = (r1*r2+rj*rj1) * disp_frac[1] / (2*dr*r1*r2);
			constnumber = del_z*del_r*del_phi/12.0;
			mid_wu[0] = iw_mid.wu[0];
			mid_wu[1] = (r1-rj)*(rj1+r1)/(4*dr*r1)+(r2-rj)*(rj1+r2)/(4*dr*r2);
			mid_wu[2] = iw_mid.wu[2];
			mid_wl[0] = iw_mid.wl[0];
			mid_wl[1] = 1.0 - mid_wu[1];
			mid_wl[2] = iw_mid.wl[2];
			
			d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_z*(1-mid_wu[1]-mid_wu[2]+mid_wu[1]*mid_wu[2])+constnumber)*q2dt;
			d_Jz[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_z*(1-mid_wu[1])*mid_wu[2]-constnumber)*q2dt;
			d_Jz[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt;
			d_Jz[k1*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt;
			d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt;
			d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt;
			d_Jr[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt;
			d_Jr[k1*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i] += (del_phi*(1-mid_wu[0]-mid_wu[1]+mid_wu[0]*mid_wu[1])+constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + j*(n_cell_z+1) + i+1] += (del_phi*mid_wu[0]*(1-mid_wu[1])-constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i] += (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt;
			d_Jphi[k*(n_cell_r+1)*(n_cell_z+1) + (j+1)*(n_cell_z+1) + i+1] += (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt;

			// Jz.AddValue(i, j, k, (del_z*(1-mid_wu[1]-mid_wu[2]+mid_wu[1]*mid_wu[2])+constnumber)*q2dt );
			// Jz.AddValue(i, j, k1, (del_z*(1-mid_wu[1])*mid_wu[2]-constnumber)*q2dt );
			// Jz.AddValue(i, j + 1, k, (del_z*mid_wu[1]*(1-mid_wu[2])-constnumber)*q2dt );
			// Jz.AddValue(i, j + 1, k1, (del_z*mid_wu[1]*mid_wu[2]+constnumber)*q2dt );
			// Jr.AddValue(i, j, k, (del_r*(1-mid_wu[0]-mid_wu[2]+mid_wu[0]*mid_wu[2])-constnumber)*q2dt );
			// Jr.AddValue(i, j, k1, (del_r*(1-mid_wu[0])*mid_wu[2]-constnumber)*q2dt );
			// Jr.AddValue(i + 1, j, k, (del_r*mid_wu[0]*(1-mid_wu[2])-constnumber)*q2dt );
			// Jr.AddValue(i + 1, j, k1, (del_r*mid_wu[0]*mid_wu[2]+constnumber)*q2dt );
			// Jphi.AddValue(i, j, k, (del_phi*(1-mid_wu[0]-mid_wu[1]+mid_wu[0]*mid_wu[1])+constnumber)*q2dt );
			// Jphi.AddValue(i + 1, j, k, (del_phi*mid_wu[0]*(1-mid_wu[1])-constnumber)*q2dt );
			// Jphi.AddValue(i, j + 1, k, (del_phi*(1-mid_wu[0])*mid_wu[1]-constnumber)*q2dt );
			// Jphi.AddValue(i + 1, j + 1, k, (del_phi*mid_wu[0]*mid_wu[1]+constnumber)*q2dt );
		}
	}
	
}

void NodeField_Cyl3D::accumulate_Rho(const IndexAndWeights_Cyl3D& iw_mid, double q){
	int i = iw_mid.indx[0];
	int j = iw_mid.indx[1];
	int k = iw_mid.indx[2];
	
	int k1 = (k+1) % m_phi_number;
	Standard_Size zrIndex[2] = {Standard_Size(i), Standard_Size(j)};
	double dz = global_grid->GetStep(0, i);
	double dr = global_grid->GetStep(1, j);
	double rj = global_grid->GetCoordComp_From_VertexVectorIndx(1,zrIndex);
	double r = rj + iw_mid.wu[1]*dr;
	double iw_wu[3], iw_wl[3];
	iw_wu[0] = iw_mid.wu[0];
	iw_wu[1] = (r-rj)*(rj+dr+r)/(2*dr*r);
	iw_wu[2] = iw_mid.wu[2];
	iw_wl[0] = iw_mid.wl[0];
	iw_wl[1] = 1.0 - iw_wu[1];
	iw_wl[2] = iw_mid.wl[2];

	double cell_vol;
	
	{ 
		if(j==1)
		{
			if(iw_mid.wu[1] <= 0.5)
			{
				iw_wu[0] = iw_mid.wu[0];
				iw_wu[1] = 4*r*(r+dr)/((2*r+dr)*(2*r+dr));
				iw_wu[2] = iw_mid.wu[2];
				iw_wl[0] = iw_mid.wl[0];
				iw_wl[1] = 1.0 - iw_wu[1];
				iw_wl[2] = iw_mid.wl[2];
			}
			for(int m_phi=0; m_phi<m_phi_number; m_phi++)
			{
				cell_vol = this->get_dual_cell_volume(i, j, m_phi);
				Rho.AddValue(i, j, m_phi, (iw_wl[0] * iw_wl[1])* q /cell_vol );
				cell_vol = this->get_dual_cell_volume(i+1, j, m_phi);
				Rho.AddValue(i + 1, j, m_phi, (iw_wu[0] * iw_wl[1])* q /cell_vol );
			}     
			cell_vol = this->get_dual_cell_volume(i, j+1, k); 
			Rho.AddValue(i, j + 1, k, (iw_wl[0] * iw_wu[1]*iw_wl[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j+1, k);  
			Rho.AddValue(i + 1, j + 1, k, (iw_wu[0] * iw_wu[1]*iw_wl[2])* q /cell_vol );	
			cell_vol = this->get_dual_cell_volume(i, j+1, k1);
			Rho.AddValue(i, j + 1, k1, (iw_wl[0] * iw_wu[1]*iw_wu[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j+1, k1);
			Rho.AddValue(i + 1, j + 1, k1, (iw_wu[0] * iw_wu[1]*iw_wu[2])* q /cell_vol );
		}
		else
		{
			cell_vol = this->get_dual_cell_volume(i, j, k);               
			Rho.AddValue(i, j, k, (iw_wl[0] * iw_wl[1]*iw_wl[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j, k); 
			Rho.AddValue(i + 1, j, k, (iw_wu[0] * iw_wl[1]*iw_wl[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i, j+1, k); 
			Rho.AddValue(i, j + 1, k, (iw_wl[0] * iw_wu[1]*iw_wl[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j+1, k);  
			Rho.AddValue(i + 1, j + 1, k, (iw_wu[0] * iw_wu[1]*iw_wl[2])* q /cell_vol );	
			cell_vol = this->get_dual_cell_volume(i, j, k1);  
			Rho.AddValue(i, j, k1, (iw_wl[0] * iw_wl[1]*iw_wu[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j, k1);  
			Rho.AddValue(i + 1, j, k1, (iw_wu[0] * iw_wl[1]*iw_wu[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i, j+1, k1);  
			Rho.AddValue(i, j + 1, k1, (iw_wl[0] * iw_wu[1]*iw_wu[2])* q /cell_vol );
			cell_vol = this->get_dual_cell_volume(i+1, j+1, k1);  
			Rho.AddValue(i + 1, j + 1, k1, (iw_wu[0] * iw_wu[1]*iw_wu[2])* q /cell_vol );
		}
	}
}






