//-----------------------------------------------------------------------------
//
// File:        Geom_TxtBuilderMap.cxx
//
// Purpose:     Construct lists of known time functions
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>


// base class includes
#include <Geom_TxtBuilderBase.hxx>


// time functors
#include <Geom_Cylinder_TxtBuilder.hxx>
#include <Geom_Polygon_TxtBuilder.hxx>
#include <Geom_Vector_TxtBuilder.hxx>
#include <Geom_Face_TxtBuilder.hxx>
#include <Geom_Revolution_TxtBuilder.hxx>

#include <Geom_SubFaceSelection_TxtBuilder.hxx>

template<> std::map<std::string, 
		    TxMakerBase< Geom_TxtBuilderBase>*, 
		    std::less<std::string> >* TxMakerMapBase< Geom_TxtBuilderBase >::makerMap=NULL;

template class TxMakerMap< Geom_TxtBuilderBase >;

TxMaker< Geom_Cylinder_TxtBuilder, Geom_TxtBuilderBase> cylBuilder("cylinder");
TxMaker< Geom_Polygon_TxtBuilder, Geom_TxtBuilderBase> polyBuilder("polygon");
TxMaker< Geom_Vector_TxtBuilder, Geom_TxtBuilderBase> vecBuilder("vector");
TxMaker< Geom_Face_TxtBuilder, Geom_TxtBuilderBase> faceBuilder("face");
TxMaker< Geom_Revolution_TxtBuilder, Geom_TxtBuilderBase> revolBuilder("revolution");



TxMaker< Geom_SubFaceSelection_TxtBuilder, Geom_TxtBuilderBase> subFaceSelection("subFaceSelection");
