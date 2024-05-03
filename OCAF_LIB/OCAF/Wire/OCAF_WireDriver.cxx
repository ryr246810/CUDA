#include <CAGDDefine.hxx>


#include "OCAF_WireDriver.ixx"
#include <OCAF_IWire.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeWire.hxx>

#include <BRepNaming_Wire.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>


#include <TDF_ChildIterator.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>

#include <TopExp_Explorer.hxx>

#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>

#include <ShapeFix_Wire.hxx>
#include <ShapeFix_Edge.hxx>

#define OK_WIRE 0
#define EMPTY_WIRE 1
#define WIRE_NOT_DONE 2
#define NULL_WIRE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_WireDriver::OCAF_WireDriver():OCAF_Driver() {}



//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_WireDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_WIRE", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_WIRE;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IWire "anInterface"
  OCAF_IWire anInterface(aNode);

  TDF_Label aResultLabel;

  TDF_AttributeMap aRefmap;
  anInterface.GetBuildWireElementsMap(aRefmap);

  Standard_Integer nbelements = aRefmap.Extent();

  TopoDS_Wire aWire;
  TopoDS_Wire theResultWire;

  BRep_Builder B;
  B.MakeWire(aWire);

  //Standard_Boolean isDone = Standard_True;

  Standard_Integer ind;

  for(ind = 1; ind<= nbelements; ind++){
    TopoDS_Shape aElement_i = anInterface.GetBuildWireElement(ind);
    if (aElement_i.IsNull()) {
      Standard_NullObject::Raise("Shape for wire construction is null");
    }
    if( aElement_i.ShapeType() == TopAbs_EDGE ) {
      B.Add( aWire, TopoDS::Edge(aElement_i) );
    }else if( aElement_i.ShapeType() == TopAbs_WIRE ) {
      TopExp_Explorer exp(aElement_i, TopAbs_EDGE);
      for( ; exp.More();exp.Next() ){
	B.Add( aWire, TopoDS::Edge(exp.Current()) );
      }
    }else{
      Standard_TypeMismatch::Raise("Shape for wire construction is neither an edge nor a wire");
    }
  }


  Handle(ShapeFix_Wire) aWireFix = new ShapeFix_Wire;
  aWireFix->Load(aWire);
  aWireFix->FixReorder();
  
  if (aWireFix->StatusReorder(ShapeExtend_FAIL1)) {
    Standard_ConstructionError::Raise("Wire construction failed: several loops detected");
  }
  else if (aWireFix->StatusReorder(ShapeExtend_FAIL)) {
    Standard_ConstructionError::Raise("Wire construction failed");
  }
  else {
  }

  // Building a Wire from unconnected edges by introducing a tolerance
  Standard_Real aTolerance = anInterface.GetTolerance();
  if (aTolerance < Precision::Confusion()){
    aTolerance = Precision::Confusion();
  }

  aWireFix->ClosedWireMode() = Standard_False;
  aWireFix->FixConnected(aTolerance);
  if (aWireFix->StatusConnected(ShapeExtend_FAIL)) {
    Standard_ConstructionError::Raise("Wire construction failed: cannot build connected wire");
  }

  if (aWireFix->StatusConnected(ShapeExtend_DONE3)) {
    aWireFix->FixGapsByRangesMode() = Standard_True;
    if (aWireFix->FixGaps3d()) {
      Handle(ShapeExtend_WireData) sbwd = aWireFix->WireData();
      Handle(ShapeFix_Edge) anEdgeFix = new ShapeFix_Edge;
      for (Standard_Integer iedge = 1; iedge <= sbwd->NbEdges(); iedge++) {
	TopoDS_Edge aEdge = TopoDS::Edge(sbwd->Edge(iedge));
	anEdgeFix->FixVertexTolerance(aEdge);
	anEdgeFix->FixSameParameter(aEdge);
      }
    }
    else if (aWireFix->StatusGaps3d(ShapeExtend_FAIL)) {
      Standard_ConstructionError::Raise("Wire construction failed: cannot fix 3d gaps");
    }
  }

  theResultWire = aWireFix->WireAPIMake();

  // 5.check the result wire
  if ( theResultWire.IsNull() ) return NULL_WIRE;
  if ( !BRepAlgo::IsValid(theResultWire) ) return WIRE_NOT_DONE;    //???????????????

  /**********************************************************************************/
  // modified by wangyue 2012.10.30 10.21(AM)
  Standard_Integer anOrientation = anInterface.GetOrientation();
  if(anOrientation == 1) {
    //theResultWire.Orientation(TopAbs_FORWARD);
    //theResultWire.Orientation(TopAbs_FORWARD);
  }
  else if(anOrientation == 2){
    //theResultWire.Orientation(TopAbs_REVERSED);
    //theResultWire = theResultWire.Reversed();
    theResultWire.Reverse();
  }
  /*********************************************************************************/

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Wire aNaming(aResultLabel);
  aNaming.Load(theResultWire, BRepNaming_WIRE);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_WIRE;
}

