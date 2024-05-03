
#include <TxMaker.h>
#include <TxMakerMap.h>

#include <STFunc.hxx>

#include <ZRTExpression.hxx>
#include <STExpression.hxx>
#include <STConstantFunc.hxx>
//
// All of the maps for space time functors
//

template <> std::map<std::string, TxMakerBase< STFunc >*, std::less<std::string> >* TxMakerMapBase< STFunc >::makerMap=NULL;

// Instantiate the maker maps
template class TxMakerMap< STFunc >;

TxMaker< ZRTExpression, STFunc > zrExpFunc("zrtExpFunc");
TxMaker< STExpression, STFunc > xyzExpFunc("xyztExpFunc");
TxMaker< STConstantFunc, STFunc > constantFunc("constant");

