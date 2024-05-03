//-----------------------------------------------------------------------------
//
// File:        Dynamic_ComboEMFields_Map.cxx
//
// Purpose:     Construct lists of known time functions
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>


// base class includes
#include <Dynamic_ComboEMFieldsBase.hxx>


// time functors
#include <SI_SC_ComboEMFields.hxx>
#include <SI_SC_ComboEMFields_Cyl3D.hxx>
#include "SI_SC_Matrix_EMFields_Cyl3D.hxx"

template<> std::map<std::string, 
		    TxMakerBase< Dynamic_ComboEMFieldsBase>*, 
		    std::less<std::string> >* TxMakerMapBase< Dynamic_ComboEMFieldsBase >::makerMap=NULL;

template class TxMakerMap< Dynamic_ComboEMFieldsBase >;

TxMaker< SI_SC_ComboEMFields, Dynamic_ComboEMFieldsBase> siSCFIT("SI_SC");

TxMaker< SI_SC_ComboEMFields_Cyl3D, Dynamic_ComboEMFieldsBase> siSCFIT_Cyl3D("SI_SC_Cyl3D");

TxMaker< SI_SC_Matrix_EMFields_Cyl3D, Dynamic_ComboEMFieldsBase> siSCFIT_Matrix_Cyl3D("SI_SC_Matrix_Cyl3D");
