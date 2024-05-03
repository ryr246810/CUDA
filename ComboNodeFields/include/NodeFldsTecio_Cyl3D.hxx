#ifndef _NodeFldsTecio_Cyl3D_HeaderFile
#define _NodeFldsTecio_Cyl3D_HeaderFile

#include <string>
#include <DynObj.hxx>
#include <ComboFieldsDefineRules.hxx>
#include <GridGeometry.hxx>
#include <IndexAndWeights.hxx>

#include <NodeFlds_OutputBase.hxx>
#include <Dynamic_ComboEMFieldsBase.hxx>
#include <TECIO.h>
#include <../Ptcl_Cyl3D/NodeField_Cyl3D.cuh>

class NodeFldsTecio_Cyl3D
{	
public:
	NodeFldsTecio_Cyl3D();
	NodeFldsTecio_Cyl3D(GridGeometry_Cyl3D* gridGeom_Cyl3D, ComboFieldsDefineRules* theComboFldDefRules, 
	Dynamic_ComboEMFieldsBase* theEMFields, NodeField_Cyl3D* node_fld_Cyl3D);
	virtual void Tecio_Mag_ZR(Standard_Integer dir, string file_name);
	virtual void Tecio_Mag_Phi(string file_name);
	virtual void Tecio_Elec_ZR(Standard_Integer dir, string file_name);
	virtual void Tecio_Elec_Phi(string file_name);
	virtual void Tecio_rphiFacet_Elec_Phi(string file_name, Standard_Real zLocation, Standard_Integer index);
	virtual void ZeroMagZRDatas();
	virtual void ZeroMagPhiDatas();
	virtual void ZeroElecZRDatas();
	virtual void ZeroElecPhiDatas();
	virtual void ZerorphiFacetElecPhiDatas();
	
	virtual void GetTheDirPhysRgnEdgeDatas(Standard_Integer edgeDir, vector<GridEdgeData*>& theEdgeDatas, const GridGeometry* gridGeom);
	virtual void TECIO_OutputAllFldDatas(string file_name);
	virtual void ZeroAllTECIODatas();
	
	
private:
	NodeField_Cyl3D* m_node_fld;
	Standard_Integer rDirEdgeNum, zDirEdgeNum;
	Standard_Integer FaceNumInOnePhiFacet;
	
	GridGeometry_Cyl3D* m_GridGeom_Cyl3D;
	Dynamic_ComboEMFieldsBase* m_theEMFields;
	ComboFieldsDefineRules* m_ComboFldDefRules;
	vector<GridEdgeData*> m_zDirGridEdgeDatas;
	vector<GridEdgeData*> m_rDirGridEdgeDatas;
	vector<vector<double>> m_zDirElec, m_rDirElec, m_zDirMag, m_rDirMag, m_PhiElec, m_PhiMag, m_rphiFacetPhiElec;
};


#endif





