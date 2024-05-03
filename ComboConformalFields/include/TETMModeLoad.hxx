#ifndef _TETMModeLoad_HeaderFile
#define _TETMModeLoad_HeaderFile

#include <string>
#include <DynObj.hxx>
#include <ComboFieldsDefineRules.hxx>
#include <ComboFields_Dynamic_SrcBase.hxx>
#include <TFunc.hxx>
#include <TxMakerMap.h>
#include <TxMakerMapBase.h>
#include <TxHierAttribSet.h>
#include <PhysConsts.hxx>
#include <GridEdge.hxx>

class TETMModeLoad : public ComboFields_Dynamic_SrcBase{
public:
	TETMModeLoad();
	~TETMModeLoad();
	
	void SetAttrib(const TxHierAttribSet& tha);
	
	void Setup();
	
	void SetupOneColumnrDirEdge();
	void SetupOneColumnzDirEdge();
	
	void Advance()
	{
		DynObj::Advance();
	}
	
	void Advance_SI_J(const double si_scale);
	void Advance_SI_MJ(const double si_scale);

private:
	TFunc* m_tfuncPtr;
	string m_ModeName;
	int m_ModeNum;
	double m_freq;
	double m_amplitude, pmn;
	double R;
	const GridGeometry_Cyl3D* m_GridGeometry_Cyl3D;
	
	vector<GridEdgeData*> m_rDirEdgeDatas;
	vector<GridEdgeData*> m_zDirEdgeDatas;
	vector<GridEdgeData*> m_GridEdgeDatas;

};




#endif
