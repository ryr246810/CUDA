//-----------------------------------------------------------------------------
// File:        PtclSourceMap.cpp
//
// Purpose:     Construct the makers for particle sources.
//-----------------------------------------------------------------------------

#include <TxMaker.h>
#include <TxMakerMap.h>
#include <TxMakerMapBase.h>

#include <Grid_Tool.hxx>
#include <PadeGrid_Tool.hxx>
#include <UniformGrid_Tool.hxx>

#include <string>

//
// All of the functor with no arguments
//

/*
template <class B> 
std::map<std::string, TxMakerBase< B >*, std::less<std::string> >* TxMakerMapBase< B >::makerMap=NULL;
//*/

template<> std::map<std::string, TxMakerBase< Grid_Tool >*, std::less<std::string> >* TxMakerMapBase< Grid_Tool >::makerMap=NULL;


template class TxMakerMap< Grid_Tool >;


TxMaker< PadeGrid_Tool, Grid_Tool > padeGrid("padeGrid");
TxMaker< UniformGrid_Tool, Grid_Tool > uniformGrid("uniformGrid");
