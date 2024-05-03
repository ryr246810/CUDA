//-----------------------------------------------------------------------------
//
// File:        TFuncMap.cxx
//
// Purpose:     Construct lists of known time functions
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>


// base class includes
#include <TFunc.hxx>


// time functors
#include <TTrapezePulse.hxx>
#include <TRisingPulse.hxx>

#include <TModulatedTrapezePulse.hxx>

#include <TGaussianEnvelopePulse.hxx>

#include <TExpression.hxx>
#include <TConstantFunc.hxx>


template<> std::map<std::string, TxMakerBase< TFunc >*, std::less<std::string> >* TxMakerMapBase< TFunc >::makerMap=NULL;

template class TxMakerMap< TFunc >;


TxMaker< TTrapezePulse, TFunc> trapezePulseFunc("TrapezePulseFunc");
TxMaker< TRisingPulse, TFunc> risingPulseFunc("RisingPulseFunc");

TxMaker< TModulatedTrapezePulse, TFunc> modulatedTrapezePulseFunc("MolulatedTrapezePulseFunc");

TxMaker< TGaussianEnvelopePulse, TFunc> gaussianEnvelopePulseFunc("GaussianEnvelopePulseFunc");

TxMaker< TConstantFunc, TFunc> constantTFunc("ConstantFunc");
TxMaker< TExpression, TFunc> expressionTFunc("ExpressionFunc");
