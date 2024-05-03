#include <NodeFldsTecio_Cyl3D.hxx>
#include <sys/types.h>
#include <sys/stat.h>
#include <Dynamic_ComboEMFieldsBase.hxx>
#include <PhysConsts.hxx>

NodeFldsTecio_Cyl3D::NodeFldsTecio_Cyl3D(GridGeometry_Cyl3D* gridGeom_Cyl3D, ComboFieldsDefineRules* theComboFldDefRules, Dynamic_ComboEMFieldsBase* theEMFields, NodeField_Cyl3D* node_fld_Cyl3D)
{
	m_GridGeom_Cyl3D = gridGeom_Cyl3D;
	m_node_fld = node_fld_Cyl3D;
	m_ComboFldDefRules = theComboFldDefRules;
	FaceNumInOnePhiFacet = m_GridGeom_Cyl3D->GetFaceSize();
	Standard_Integer TotalZEdgeNum = m_GridGeom_Cyl3D->GetEdgeSize(0);
	Standard_Integer TotalREdgeNum = m_GridGeom_Cyl3D->GetEdgeSize(1);
	rDirEdgeNum = TotalREdgeNum - FaceNumInOnePhiFacet;
	zDirEdgeNum = FaceNumInOnePhiFacet / rDirEdgeNum;
	m_theEMFields = theEMFields;
	Standard_Integer phiNumber = gridGeom_Cyl3D->GetDimPhi();
	
	m_rDirElec.resize(zDirEdgeNum+1);
	m_rDirMag.resize(zDirEdgeNum);
	m_PhiElec.resize(zDirEdgeNum+1);
	m_rphiFacetPhiElec.resize(2*rDirEdgeNum+1);
	for(int i=0; i<m_rDirElec.size(); i++)
	{
		m_rDirElec[i].resize(rDirEdgeNum);
	}
	for(int i=0; i<m_rDirMag.size(); i++)
	{
		m_rDirMag[i].resize(rDirEdgeNum+1);
	}
	for(int i=0; i<m_PhiElec.size(); i++)
	{
		m_PhiElec[i].resize(rDirEdgeNum+1);
	}
	
	for(int i=0; i<m_rphiFacetPhiElec.size(); i++)
	{
		m_rphiFacetPhiElec[i].resize(2*rDirEdgeNum+1);
	}
	
	m_zDirElec.resize(zDirEdgeNum);
	m_zDirMag.resize(zDirEdgeNum+1);
	m_PhiMag.resize(zDirEdgeNum);
	for(int i=0; i<m_zDirElec.size(); i++)
	{
		m_zDirElec[i].resize(rDirEdgeNum+1);
	}
	for(int i=0; i<m_zDirMag.size(); i++)
	{
		m_zDirMag[i].resize(rDirEdgeNum);
	}
	for(int i=0; i<m_PhiMag.size(); i++)
	{
		m_PhiMag[i].resize(rDirEdgeNum);
	}
}

void NodeFldsTecio_Cyl3D::TECIO_OutputAllFldDatas(string file_name)
{
	Tecio_Elec_ZR(0,file_name);
	Tecio_Elec_ZR(1,file_name);
	Tecio_Elec_Phi(file_name);
	Tecio_Mag_ZR(0,file_name);
	Tecio_Mag_ZR(1,file_name);
	Tecio_Mag_Phi(file_name);
}

void NodeFldsTecio_Cyl3D::ZeroAllTECIODatas()
{
	ZeroElecZRDatas();
	ZeroElecPhiDatas();
	ZeroMagZRDatas();
	ZeroMagPhiDatas();
}

void NodeFldsTecio_Cyl3D::GetTheDirPhysRgnEdgeDatas(Standard_Integer edgeDir, vector<GridEdgeData*>& theEdgeDatas, const GridGeometry* gridGeom)
{
	vector<GridEdgeData*> tmptheEdgeDatas;
	gridGeom->GetAllGridEdgeDatasOfPhysRgn(false, tmptheEdgeDatas);
	for(int i=0; i<tmptheEdgeDatas.size(); i++)
	{
		if(tmptheEdgeDatas[i]->GetDir() != edgeDir) continue;
		else theEdgeDatas.push_back(tmptheEdgeDatas[i]);
	}
}


void NodeFldsTecio_Cyl3D::ZeroMagZRDatas()
{
	for(int i=0; i<m_zDirMag.size(); i++)
		for(int j=0; j<m_zDirMag[0].size(); j++)
		{
			m_zDirMag[i][j] = 0.0;
		}
		
	for(int i=0; i<m_rDirMag.size(); i++)
		for(int j=0; j<m_rDirMag[0].size(); j++)
		{
			m_rDirMag[i][j] = 0.0;
		}
}

void NodeFldsTecio_Cyl3D::ZeroMagPhiDatas()
{
	for(int i=0; i<m_PhiMag.size(); i++)
		for(int j=0; j<m_PhiMag[0].size(); j++)
		{
			m_PhiMag[i][j] = 0.0;
		}
}

void NodeFldsTecio_Cyl3D::ZeroElecZRDatas()
{
	for(int i=0; i<m_zDirElec.size(); i++)
		for(int j=0; j<m_zDirElec[0].size(); j++)
		{
			m_zDirElec[i][j] = 0.0;
		}
	
	for(int i=0; i<m_rDirElec.size(); i++)
		for(int j=0; j<m_rDirElec[0].size(); j++)
		{
			m_rDirElec[i][j] = 0.0;
		}
}

void NodeFldsTecio_Cyl3D::ZeroElecPhiDatas()
{
	for(int i=0; i<m_PhiElec.size(); i++)
		for(int j=0; j<m_PhiElec[0].size(); j++)
		{
			m_PhiElec[i][j] = 0.0;
		}
}

void NodeFldsTecio_Cyl3D::ZerorphiFacetElecPhiDatas()
{
	for(int i=0; i<m_rphiFacetPhiElec.size(); i++)
		for(int j=0; j<m_rphiFacetPhiElec[0].size(); j++)
		{
			m_rphiFacetPhiElec[i][j] = 0.0;
		}
}

void NodeFldsTecio_Cyl3D::Tecio_Mag_ZR(Standard_Integer dir, string file_name)
{
	Standard_Integer edgedir = (dir+1)%2;
	Standard_Integer phiNumber = m_GridGeom_Cyl3D->GetDimPhi();
	for(int k=0; k<phiNumber; k++)
	{
		const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(k);
		if(dir == 0) 
		{
			string HzOutputDirName = file_name+"/Tecio_Hz";
			int flag = mkdir(HzOutputDirName.c_str(), 0777);
			stringstream ss;
			ss<<"Hz"<<",phi="<<k<<".txt";
			HzOutputDirName += "/"+ss.str();
			fstream fout;
			fout.open(HzOutputDirName.c_str(), ios::out);
			fout<<"TITLE="<<'"'<<"Hz"<<'"'<<'\n';
			fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"Hz"<<'"'<<'\n';
			ZeroMagZRDatas();
			vector<GridEdgeData*> theEdgeDatas;
			theEdgeDatas.clear();
			GetTheDirPhysRgnEdgeDatas(edgedir, theEdgeDatas, gridGeom);
			for(int i=0; i<theEdgeDatas.size(); i++)
			{
				Standard_Integer currEdgeIndex = theEdgeDatas[i]->GetBaseGridEdge()->GetIndex();
				Standard_Integer dynMFldIndx = m_ComboFldDefRules->Get_DynamicMagField_PhysDataIndex();
				m_zDirMag[currEdgeIndex/rDirEdgeNum][currEdgeIndex%rDirEdgeNum] = theEdgeDatas[i]->GetSweptPhysData(dynMFldIndx);
			}
			for(int i=0; i<zDirEdgeNum; i++)
			{
				fout<<"ZONE T="<<'"'<<"Bz"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
				fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum; j++)
				{
					fout<<m_zDirMag[i][j]<<" ";
				}
				fout<<"\n\n";
			}
			fout.close();
		}
		if(dir == 1)
		{
			string HrOutputDirName = file_name+"/Tecio_Hr";
			int flag = mkdir(HrOutputDirName.c_str(), 0777);
			stringstream ss;
			ss<<"Hr"<<",phi="<<k<<".txt";
			HrOutputDirName += "/"+ss.str();
			fstream fout;
			fout.open(HrOutputDirName.c_str(), ios::out);
			fout<<"TITLE="<<'"'<<"Hr"<<'"'<<'\n';
			fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"Hr"<<'"'<<'\n';
			ZeroMagZRDatas();
			vector<GridEdgeData*> theEdgeDatas;
			theEdgeDatas.clear();
			GetTheDirPhysRgnEdgeDatas(edgedir, theEdgeDatas, gridGeom);
			for(int i=0; i<theEdgeDatas.size(); i++)
			{
				Standard_Integer currEdgeIndex = theEdgeDatas[i]->GetBaseGridEdge()->GetIndex();
				Standard_Integer dynMFldIndx = m_ComboFldDefRules->Get_DynamicMagField_PhysDataIndex();
				m_rDirMag[currEdgeIndex/(rDirEdgeNum+1)][currEdgeIndex%(rDirEdgeNum+1)] = theEdgeDatas[i]->GetSweptPhysData(dynMFldIndx);
			}
			for(int i=0; i<zDirEdgeNum; i++)
			{
				fout<<"ZONE T="<<'"'<<"Br"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
				fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double rnode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<rnode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double rnode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<rnode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum; j++)
				{
					fout<<m_rDirMag[i][j]<<" ";
				}
				fout<<"\n\n";
			}
			fout.close();
		}
	}
	
}

void NodeFldsTecio_Cyl3D::Tecio_Mag_Phi(string file_name)
{
	Standard_Integer phiNumber = m_GridGeom_Cyl3D->GetDimPhi();
	for(int k=0; k<phiNumber; k++)
	{
		const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(k);
		string HPhiOutputDirName = file_name+"/Tecio_HPhi";
		int flag = mkdir(HPhiOutputDirName.c_str(), 0777);
		stringstream ss;
		ss<<"HPhi"<<",phi="<<k<<".txt";
		HPhiOutputDirName += "/"+ss.str();
		fstream fout;
		fout.open(HPhiOutputDirName.c_str(), ios::out);
		fout<<"TITLE="<<'"'<<"HPhi"<<'"'<<'\n';
		fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"HPhi"<<'"'<<'\n';
		ZeroMagPhiDatas();
		vector<GridFaceData*> theFaceDatas;
		theFaceDatas.clear();
		gridGeom->GetAllGridFaceDatasOfPhysRgn(theFaceDatas);
		for(int i=0; i<theFaceDatas.size(); i++)
		{
			Standard_Integer currFaceIndex = theFaceDatas[i]->GetBaseGridFace()->GetIndex();
			Standard_Integer dynMFldIndx = m_ComboFldDefRules->Get_DynamicMagField_PhysDataIndex();
			m_PhiMag[currFaceIndex/rDirEdgeNum][currFaceIndex%rDirEdgeNum] = theFaceDatas[i]->GetPhysData(dynMFldIndx);
		}

		for(int i=0; i<zDirEdgeNum; i++)
		{
			fout<<"ZONE T="<<'"'<<"Bphi"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
			fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
				fout<<znode<<" ";
			}
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
				fout<<znode<<" ";
			}
			fout<<"\n\n";
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double rnode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
				fout<<rnode<<" ";
			}
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double rnode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
				fout<<rnode<<" ";
			}
			fout<<"\n\n";
			for(int j=0; j<rDirEdgeNum; j++)
			{
				fout<<m_PhiMag[i][j]<<" ";
			}
			fout<<"\n\n";
		}
		fout.close();
	}
}

void NodeFldsTecio_Cyl3D::Tecio_Elec_ZR(Standard_Integer edgedir, string file_name)
{
	Standard_Integer phiNumber = m_GridGeom_Cyl3D->GetDimPhi();
	for(int k=0; k<phiNumber; k++)
	{
		const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(k);
		if(edgedir == 0) 
		{
			string EzOutputDirName = file_name+"/Tecio_Ez";
			int flag = mkdir(EzOutputDirName.c_str(), 0777);
			stringstream ss;
			ss<<"Ez"<<",phi="<<k<<".txt";
			EzOutputDirName += "/"+ss.str();
			fstream fout;
			fout.open(EzOutputDirName.c_str(), ios::out);
			fout<<"TITLE="<<'"'<<"Ez"<<'"'<<'\n';
			fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"Ez"<<'"'<<'\n';
			ZeroElecZRDatas();
			vector<GridEdgeData*> theEdgeDatas;
			theEdgeDatas.clear();
			GetTheDirPhysRgnEdgeDatas(edgedir, theEdgeDatas, gridGeom);
			for(int i=0; i<theEdgeDatas.size(); i++)
			{
				Standard_Integer currEdgeIndex = theEdgeDatas[i]->GetBaseGridEdge()->GetIndex();
				Standard_Integer dynEFldIndx = m_ComboFldDefRules->Get_DynamicElecField_PhysDataIndex();
				m_zDirElec[currEdgeIndex/(rDirEdgeNum+1)][currEdgeIndex%(rDirEdgeNum+1)] = theEdgeDatas[i]->GetPhysData(dynEFldIndx);
			}
			for(int i=0; i<zDirEdgeNum; i++)
			{
				fout<<"ZONE T="<<'"'<<"Ez"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
				fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum; j++)
				{
					fout<<m_zDirElec[i][j]<<" ";
				}
				fout<<"\n\n";
			}
			fout.close();
		}
		if(edgedir == 1)
		{
			string ErOutputDirName = file_name+"/Tecio_Er";
			int flag = mkdir(ErOutputDirName.c_str(), 0777);
			stringstream ss;
			ss<<"Er"<<",phi="<<k<<".txt";
			ErOutputDirName += "/"+ss.str();
			fstream fout;
			fout.open(ErOutputDirName.c_str(), ios::out);
			fout<<"TITLE="<<'"'<<"Er"<<'"'<<'\n';
			fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"Er"<<'"'<<'\n';
			ZeroElecZRDatas();
			vector<GridEdgeData*> theEdgeDatas;
			theEdgeDatas.clear();
			GetTheDirPhysRgnEdgeDatas(edgedir, theEdgeDatas, gridGeom);
			for(int i=0; i<theEdgeDatas.size(); i++)
			{
				Standard_Integer currEdgeIndex = theEdgeDatas[i]->GetBaseGridEdge()->GetIndex();
				Standard_Integer dynEFldIndx = m_ComboFldDefRules->Get_DynamicElecField_PhysDataIndex();
				m_rDirElec[currEdgeIndex/rDirEdgeNum][currEdgeIndex%rDirEdgeNum] = theEdgeDatas[i]->GetPhysData(dynEFldIndx);
			}
			for(int i=0; i<zDirEdgeNum; i++)
			{
				fout<<"ZONE T="<<'"'<<"Er"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
				fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
					fout<<znode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double rnode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<rnode<<" ";
				}
				for(int j=0; j<rDirEdgeNum+1; j++)
				{
					double rnode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
					fout<<rnode<<" ";
				}
				fout<<"\n\n";
				for(int j=0; j<rDirEdgeNum; j++)
				{
					fout<<m_rDirElec[i][j]<<" ";
				}
				fout<<"\n\n";
			}
			fout.close();
		}
	}
	
}

void NodeFldsTecio_Cyl3D::Tecio_Elec_Phi(string file_name)
{
	Standard_Integer phiNumber = m_GridGeom_Cyl3D->GetDimPhi();
	for(int k=0; k<phiNumber; k++)
	{
		const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(k);
		string HPhiOutputDirName = file_name+"/Tecio_EPhi";
		int flag = mkdir(HPhiOutputDirName.c_str(), 0777);
		stringstream ss;
		ss<<"EPhi"<<",phi="<<k<<".txt";
		HPhiOutputDirName += "/"+ss.str();
		fstream fout;
		fout.open(HPhiOutputDirName.c_str(), ios::out);
		fout<<"TITLE="<<'"'<<"EPhi"<<'"'<<'\n';
		fout<<"VARIABLES="<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"EPhi"<<'"'<<'\n';
		ZeroElecPhiDatas();
		vector<GridVertexData*> theGridVertexDatas;
		theGridVertexDatas.clear();
		//theGridVertexDatas = m_theEMFields->GetCntrElecVertices();
		gridGeom->GetAllGridVertexDatasOfPhysRgn(false, theGridVertexDatas);
		for(int i=0; i<theGridVertexDatas.size(); i++)
		{
			Standard_Integer currVertexIndex = theGridVertexDatas[i]->GetIndex();
			Standard_Integer dynEFldIndx = m_ComboFldDefRules->Get_DynamicElecField_PhysDataIndex();
			m_PhiElec[currVertexIndex/(rDirEdgeNum+1)][currVertexIndex%(rDirEdgeNum+1)] = theGridVertexDatas[i]->GetSweptPhysData(dynEFldIndx);
		}
		for(int i=0; i<zDirEdgeNum; i++)
		{
			fout<<"ZONE T="<<'"'<<"Ephi"<<'"'<<','<<" "<<"I="<<rDirEdgeNum+1<<", "<<"J="<<2<<", ";
			fout<<"DATAPACKING=BLOCK"<<endl;
			//the first
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double znode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[0];
				fout<<znode<<" ";
			}
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double znode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[0];
				fout<<znode<<" ";
			}
			fout<<"\n\n";
			//the first
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double rnode = (gridGeom->GetGridVertices()+i*(rDirEdgeNum+1) + j)->GetLocation()[1];
				fout<<rnode<<" ";
			}
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				double rnode = (gridGeom->GetGridVertices()+(i+1)*(rDirEdgeNum+1) + j)->GetLocation()[1];
				fout<<rnode<<" ";
			}
			fout<<"\n\n";
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				fout<<m_PhiElec[i][j]<<" ";
			}
			for(int j=0; j<rDirEdgeNum+1; j++)
			{
				fout<<m_PhiElec[i+1][j]<<" ";
			}
			fout<<"\n\n";
		}
		fout.close();
	}
}


void NodeFldsTecio_Cyl3D::Tecio_rphiFacet_Elec_Phi(string file_name, Standard_Real zLocation, Standard_Integer index)
{
	int phiDivNumber = 360;
	string HPhiOutputDirName = file_name+"/Tecio_rPhiFacetEPhi";
	int flag = mkdir(HPhiOutputDirName.c_str(), 0777);
	stringstream ss;
	ss<<"EPhi"<<index<<".txt";
	HPhiOutputDirName += "/"+ss.str();
	fstream fout;
	fout.open(HPhiOutputDirName.c_str(), ios::out);
	fout<<"TITLE="<<'"'<<"EPhi"<<'"'<<'\n';
	fout<<"VARIABLES="<<'"'<<"x"<<'"'<<", "<<'"'<<"y"<<'"'<<", "<<'"'<<"EPhi"<<'"'<<'\n';
	Standard_Size zIndex;
	m_GridGeom_Cyl3D->GetZRGrid()->ComputeIndex(0,zLocation,zIndex);
	double dr = m_GridGeom_Cyl3D->GetZRGrid()->GetStep(1,1);
	Standard_Integer phiNumber = m_GridGeom_Cyl3D->GetDimPhi();
	ZerorphiFacetElecPhiDatas();
	const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(0);
	for(int k=0; k<phiDivNumber; k++)
	{
		int k1 = (k+1)%phiDivNumber;
		ZeroElecPhiDatas();
		vector<GridVertexData*> theGridVertexDatas;
		theGridVertexDatas.clear();
		set<Standard_Integer> theBndsDefine; 
		m_theEMFields->GetFldsDefCntr()->GetFieldsDefineRules()->GetBndElecMaterialSet(theBndsDefine);
		//gridGeom->GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(theBndsDefine,false,theGridVertexDatas);
		gridGeom->GetAllGridVertexDatasOfPhysRgn(false, theGridVertexDatas);
		Standard_Size zrIndex[2];
		int DataPoint = 0;
		vector<GridVertexData*> GridVertexDatasOnOnezFacet;
		GridVertexDatasOnOnezFacet.clear();
		for(int i=0; i<theGridVertexDatas.size(); i++)
		{
			theGridVertexDatas[i]->GetVecIndex(zrIndex);
			if(zrIndex[0] == zIndex)
			{
				DataPoint++;
				GridVertexDatasOnOnezFacet.push_back(theGridVertexDatas[i]);
			}
		}
		fout<<"ZONE T="<<'"'<<"Ephi"<<'"'<<','<<" "<<"I="<<DataPoint<<", "<<"J="<<2<<", ";
		fout<<"DATAPACKING=POINT"<<"\n";
		for(int j=0; j<DataPoint; j++)
		{
			TxVector<double> pos, e_field;
			pos[0] = zLocation;
			pos[1] = GridVertexDatasOnOnezFacet[j]->GetLocation()[1];
			pos[2] = 2*mksConsts.pi*k/phiDivNumber;
			m_node_fld->fill_with_E(pos, e_field);
			fout<<pos[1]*cos(pos[2])<<" ";
			fout<<pos[1]*sin(pos[2])<<" ";
			fout<<e_field[2]<<"\n";
		}
		for(int j=0; j<DataPoint; j++)
		{
			TxVector<double> pos, e_field;
			pos[0] = zLocation;
			pos[1] = GridVertexDatasOnOnezFacet[j]->GetLocation()[1];
			pos[2] = 2*mksConsts.pi*k1/phiDivNumber;
			m_node_fld->fill_with_E(pos, e_field);
			fout<<pos[1]*cos(pos[2])<<" ";
			fout<<pos[1]*sin(pos[2])<<" ";
			fout<<e_field[2]<<"\n";
		}
	}
	fout.close();
}

/*
void NodeFldsTecio_Cyl3D::Tecio_rphiFacet_Elec_Phi(string file_name, Standard_Real zLocation)
{
	int phiDivNumber = 72;
	string HPhiOutputDirName = file_name+"/Tecio_rPhiFacetEPhi";
	int flag = mkdir(HPhiOutputDirName.c_str(), 0777);
	stringstream ss;
	ss<<"EPhi"<<".txt";
	HPhiOutputDirName += "/"+ss.str();
	fstream fout;
	fout.open(HPhiOutputDirName.c_str(), ios::out);
	Standard_Size zIndex;
	m_GridGeom_Cyl3D->GetZRGrid()->ComputeIndex(0,zLocation,zIndex);
	ZerorphiFacetElecPhiDatas();
	const GridGeometry* gridGeom = m_GridGeom_Cyl3D->GetGridGeometry(0);
	for(int k=0; k<phiDivNumber; k++)
	{
		ZeroElecPhiDatas();
		vector<GridVertexData*> theGridVertexDatas;
		theGridVertexDatas.clear();
		gridGeom->GetAllGridVertexDatasOfPhysRgn(true, theGridVertexDatas);
		for(int i=0; i<theGridVertexDatas.size(); i++)
		{
			Standard_Size zrIndex[2];
			theGridVertexDatas[i]->GetVecIndex(zrIndex);
			if(zrIndex[0] == zIndex)
			{
				fout<<theGridVertexDatas[i]->GetLocation()[1]*cos(2*mksConsts.pi*k/phiDivNumber)<<"\t";
				fout<<theGridVertexDatas[i]->GetLocation()[1]*sin(2*mksConsts.pi*k/phiDivNumber)<<"\t";
				TxVector<double> pos, e_field;
				pos[0] = zLocation; 
				pos[1] = theGridVertexDatas[i]->GetLocation()[1];
				pos[2] = 2*mksConsts.pi*k/phiDivNumber;
				m_node_fld->fill_with_E(pos, e_field);
				fout<<e_field[2]<<"\n";
			}
		}
	}
	fout.close();
}
*/





