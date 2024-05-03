//-----------------------------------------------------------------------------
//
// File:        FieldsDgnMap.cxx
//
// Purpose:     base dgn declaration
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>

// base class includes
#include <FieldsDgnBase.hxx>

#include <DynFlds_VertexData_Dgn.hxx>
#include <DynFlds_EdgeData_Dgn.hxx>
#include <DynFlds_FaceData_Dgn.hxx>
#include <DynFlds_SweptEdgeData_Dgn.hxx>
#include <DynFlds_PoyntingFlux_Dgn.hxx>
#include <DynFlds_Voltage_Dgn.hxx>
#include <DynFlds_Current_Dgn.hxx>


template<> std::map<std::string, 
		    TxMakerBase< FieldsDgnBase>*, 
		    std::less<std::string> >* TxMakerMapBase< FieldsDgnBase >::makerMap=NULL;

template class TxMakerMap< FieldsDgnBase >;

TxMaker< DynFlds_EdgeData_Dgn, FieldsDgnBase> edgeElecDgn("ElecDgn_ZR");
TxMaker< DynFlds_VertexData_Dgn, FieldsDgnBase> vertexElecDgn("ElecDgn_Phi");
TxMaker< DynFlds_FaceData_Dgn, FieldsDgnBase> faceMagDgn("MagDgn_Phi");

TxMaker< DynFlds_SweptEdgeData_Dgn, FieldsDgnBase> sweptFaceElecDgn("MagDgn_ZR");

TxMaker< DynFlds_PoyntingFlux_Dgn, FieldsDgnBase> poyntingDgn("PoyntingDgn"); // 能流密度

TxMaker< DynFlds_Voltage_Dgn, FieldsDgnBase> voltageDgn("VoltageDgn");

TxMaker< DynFlds_Current_Dgn, FieldsDgnBase> currentDgn("CurrentDgn");
