//-----------------------------------------------------------------------------
//
// File:        ComboFields_Dynamic_SrcMap.cxx
//
// Purpose:     Construct lists of known time functions
//-----------------------------------------------------------------------------

// tx includes
#include <TxMaker.h>
#include <TxMakerMap.h>


// base class includes
#include <ComboFields_Dynamic_SrcBase.hxx>

#include <ComboFields_Dynamic_EdgeHardElecSrc.hxx>
#include <ComboFields_Dynamic_EdgeSoftElecSrc.hxx>

#include <ComboFields_Dynamic_VertexHardElecSrc.hxx>
#include <ComboFields_Dynamic_VertexSoftElecSrc.hxx>

#include <ComboFields_Dynamic_SweptFaceHardMagSrc.hxx>
#include <ComboFields_Dynamic_SweptFaceSoftMagSrc.hxx>

#include <ComboFields_Dynamic_MurVoltagePort.hxx>
#include <TETMModeLoad.hxx>

// 第一部分是类型，第二部分是类作用域限定，第三部分是类成员名称
template<> std::map<std::string, 
		    TxMakerBase< ComboFields_Dynamic_SrcBase>*, 
		    std::less<std::string> >* TxMakerMapBase< ComboFields_Dynamic_SrcBase >::makerMap=NULL;

template class TxMakerMap< ComboFields_Dynamic_SrcBase >;//声明


// 定义对象
TxMaker< ComboFields_Dynamic_EdgeHardElecSrc, ComboFields_Dynamic_SrcBase> edgeHardSrc("EdgeHardSrc");
TxMaker< ComboFields_Dynamic_EdgeSoftElecSrc, ComboFields_Dynamic_SrcBase> edgeSoftSrc("EdgeSoftSrc");

TxMaker< ComboFields_Dynamic_VertexHardElecSrc, ComboFields_Dynamic_SrcBase> vertexHardSrc("VertexHardSrc");
TxMaker< ComboFields_Dynamic_VertexSoftElecSrc, ComboFields_Dynamic_SrcBase> vertexSoftSrc("VertexSoftSrc");

TxMaker< ComboFields_Dynamic_SweptFaceHardMagSrc, ComboFields_Dynamic_SrcBase> sweptFaceHardSrc("SweptFaceHardSrc");
TxMaker< ComboFields_Dynamic_SweptFaceSoftMagSrc, ComboFields_Dynamic_SrcBase> sweptFaceSoftSrc("SweptFaceSoftSrc");

TxMaker< ComboFields_Dynamic_MurVoltagePort, ComboFields_Dynamic_SrcBase> murVoltagePort("MurVoltagePort");//注册
TxMaker< TETMModeLoad, ComboFields_Dynamic_SrcBase> TETMModeLoad("ModeLoad");//注册



