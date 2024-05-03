//-----------------------------------------------------------------------------
// File:         NoArgFuncMap.cpp
// Purpose:     Construct map of known no-argument functions
//-----------------------------------------------------------------------------


#include <TxMaker.h>
#include <TxMakerMap.h>
#include <TxMakerBase.h>
#include <TxMakerMapBase.h>

#include <NoArgFunc.hxx>

// No argument functors
#include <RandomFunc.hxx>

/*
template <class B>
std::map<std::string, TxMakerBase< B >* >* TxMakerMapBase< B >::makerMap = NULL;
//*/

template <> std::map<std::string, TxMakerBase< NoArgFunc >* >* TxMakerMapBase< NoArgFunc >::makerMap = NULL;


// Instantiate the maker maps
template class TxMakerMap< NoArgFunc >;

// Now make all of the functor makers.  This registers them.

TxMaker< RandomFunc, NoArgFunc > 	randomD("random");
