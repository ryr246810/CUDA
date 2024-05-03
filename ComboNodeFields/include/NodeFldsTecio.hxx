#ifndef _NodeFldsTecio_HeaderFile
#define _NodeFldsTecio_HeaderFile

#include <string>
#include <DynObj.hxx>
#include <ComboFieldsDefineRules.hxx>
#include <GridGeometry.hxx>
#include <IndexAndWeights.hxx>

#include <NodeFlds_OutputBase.hxx>
#include <Dynamic_ComboEMFieldsBase.hxx>
#include <TECIO.h>

class NodeFldsTecio
{	
public:
	NodeFldsTecio();
	NodeFldsTecio(GridGeometry* gridGeom, ComboFieldsDefineRules* theComboFldDefRules, Dynamic_ComboEMFieldsBase* theEMFields);
	virtual void Tecio_Mag_ZR(Standard_Integer dir, string file_name);
	virtual void Tecio_Mag_Phi(string file_name);
	virtual void Tecio_Elec_ZR(Standard_Integer dir, string file_name);
	virtual void Tecio_Elec_Phi(string file_name);
	virtual void ZeroMagZRDatas();
	virtual void ZeroMagPhiDatas();
	virtual void ZeroElecZRDatas();
	virtual void ZeroElecPhiDatas();
	
	virtual void GetTheDirPhysRgnEdgeDatas(Standard_Integer edgeDir, vector<GridEdgeData*>& theEdgeDatas);
	virtual void TECIO_OutputAllFldDatas(string file_name);
	virtual void ZeroAllTECIODatas();
	
private:
	Standard_Integer rDirEdgeNum, zDirEdgeNum;
	Standard_Integer FaceNumInOnePhiFacet;
	
	GridGeometry* m_GridGeom;
	Dynamic_ComboEMFieldsBase* m_theEMFields;
	ComboFieldsDefineRules* m_ComboFldDefRules;
	vector<GridEdgeData*> m_zDirGridEdgeDatas;
	vector<GridEdgeData*> m_rDirGridEdgeDatas;
	vector<vector<double>> m_zDirElec, m_rDirElec, m_zDirMag, m_rDirMag, m_PhiElec, m_PhiMag;
};


#endif





