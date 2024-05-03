//-----------------------------------------------------------------------------
//
// File:        ExprFuncMap.cxx
//
// Purpose:     Construct lists of known time functions
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>


// base class includes
#include <ExprFuncMap_Bash.hxx>

// #include <xyztExprFunc.hxx>
// #include <xyzExprFunc.hxx>
// #include <zrExprFunc.hxx>
// #include <tExprFunc.hxx>
// #include <noArgExprFunc.hxx>


template<> std::map<std::string, TxMakerBase<ExprFuncBase>*, std::less<std::string> >* TxMakerMapBase<ExprFuncBase>::makerMap=NULL;

template class TxMakerMap< ExprFuncBase >;

TxMaker< xyztExprFunc, ExprFuncBase> xyztFunc("xyztFunc");
TxMaker< xyzExprFunc, ExprFuncBase> xyzFunc("xyzFunc");
TxMaker< zrExprFunc, ExprFuncBase> zrFunc("zrFunc");
TxMaker< zrtExprFunc, ExprFuncBase> zrtFunc("zrtFunc");
TxMaker< tExprFunc, ExprFuncBase> tFunc("tFunc");
TxMaker< noArgExprFunc, ExprFuncBase> noArgFunc("noArgFunc");
