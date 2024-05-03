// File:	OCAF_FaceDriver.cxx
// Created:	2010.04.21
// Author:	Wang Yue
//		id_wangyue@hotmail.com

#include <CAGDDefine.hxx>


#include "OCAF_FaceDriver.ixx"
#include <OCAF_IFace.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>

#include <BRepNaming_Face.hxx>
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


#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>


#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <ShHealOper_ShapeProcess.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TopExp_Explorer.hxx>
#include <BRepLib_FindSurface.hxx>
#include <BRepBuilderAPI_Copy.hxx>

#include <BRepLib.hxx>

#define OK_FACE 0
#define EMPTY_FACE 1
#define FACE_NOT_DONE 2
#define NULL_FACE 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_FaceDriver::OCAF_FaceDriver():OCAF_Driver() {}



//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_FaceDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_FACE", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_FACE;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IFace "anInterface"
  OCAF_IFace anInterface(aNode);


  Standard_Integer aType = anInterface.GetType();


  TopoDS_Shape theResultShape;

  if (aType == FACE_BY_WIRE){
    TopoDS_Shape aRefShape = anInterface.GetWire();
    if (aRefShape.IsNull()){
      Standard_NullObject::Raise("Argument Shape is null");
    }
    TopoDS_Wire aRefWire;
    if(aRefShape.ShapeType() == TopAbs_WIRE) {
      aRefWire = TopoDS::Wire(aRefShape);

      TopoDS_Vertex aV1, aV2;
      TopExp::Vertices(aRefWire, aV1, aV2);
      if ( !aV1.IsNull() && !aV2.IsNull() && aV1.IsSame(aV2) ){
	aRefShape.Closed(true);
      }else{
	Standard_NullObject::Raise("Shape for face construction is not closed");
      }
    }else if(aRefShape.ShapeType() == TopAbs_EDGE){
      BRepBuilderAPI_MakeWire MW;
      MW.Add(TopoDS::Edge(aRefShape));
      if (!MW.IsDone()) {
        Standard_ConstructionError::Raise("Wire construction failed");
      }
      aRefWire = MW.Wire();
    }else{
      Standard_TypeMismatch::Raise("Wire creation aborted : the selcted shape is not a wire");
    }


    Standard_Boolean isPlanar = anInterface.GetIsPlanar();
    MakeFace(aRefWire, isPlanar, theResultShape);


  }else if(aType == FACE_BY_FACE_WIRE || aType == FACE_BY_SURFACE_WIRE){
    /*************************************************/
    TopoDS_Shape aRefShape = anInterface.GetFace();
    if (aRefShape.ShapeType() != TopAbs_FACE) {
      Standard_TypeMismatch::Raise("Wire creation aborted : the selcted shapes are not correct");
    }
    TopoDS_Face aRefFace = TopoDS::Face(aRefShape);
    Handle(Geom_Surface) aSurface = BRep_Tool::Surface(aRefFace);
    /*************************************************/
    /*************************************************/
    aRefShape = anInterface.GetWire();
    if (aRefShape.ShapeType() != TopAbs_WIRE) {
      Standard_TypeMismatch::Raise("Wire creation aborted : the selcted shape is not a wire");
    }
    TopoDS_Wire aRefWire = TopoDS::Wire(aRefShape);
    /*************************************************/
    if(aType == FACE_BY_FACE_WIRE){
      BRepBuilderAPI_MakeFace mkFace(aRefFace,aRefWire);
      mkFace.Build();
      if (!mkFace.IsDone()) return FACE_NOT_DONE;
      theResultShape = mkFace.Face();
    }
    else if(aType == FACE_BY_SURFACE_WIRE){
      BRepBuilderAPI_MakeFace mkFace(aSurface,aRefWire);
      mkFace.Build();
      if (!mkFace.IsDone()) return FACE_NOT_DONE;
      theResultShape = mkFace.Face();
    }
    /*
    ShapeFix_Face FFace(theResultShape);
    FFace.Perform();
    theResultShape = FFace.Face();
    //*/
    BRepLib::BuildCurves3d(theResultShape);
  }else{
    return EMPTY_FACE;
  }


  // 5 make a wire using the BRepNaming_Face method.
  if (theResultShape.IsNull()) return NULL_FACE;
  if ( !BRepAlgo::IsValid(theResultShape) ) return FACE_NOT_DONE;    //???????????????

  // 6. create a child label of this driver's label
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);


  // 7 append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Face aNaming(aResultLabel);
  aNaming.Load(TopoDS::Face(theResultShape), BRepNaming_FACE);
  
  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_FACE;
}

//=======================================================================
//function : MakeFace
//purpose  :
//=======================================================================
void OCAF_FaceDriver::MakeFace (const TopoDS_Wire&     theWire,
				const Standard_Boolean isPlanarWanted,
				TopoDS_Shape&          theResult) const 
{
  // try to build face on plane or on any surface under the edges of the wire
  BRepBuilderAPI_MakeFace MK (theWire, isPlanarWanted);
  if (MK.IsDone()) {
    theResult = MK.Shape();
    return;
  }
  if (!isPlanarWanted) {
    // try to construct filling surface
    BRepOffsetAPI_MakeFilling MF;

    Standard_Integer nbEdges = 0;
    BRepTools_WireExplorer aWE (theWire);
    for (; aWE.More(); aWE.Next(), nbEdges++) {
      MF.Add(TopoDS::Edge(aWE.Current()), GeomAbs_C0);
    }

    MF.Build();
    if (MF.IsDone()) {
      // Result of filling
      TopoDS_Shape aFace = MF.Shape();
      
      Handle(Geom_Surface) aGS = BRep_Tool::Surface(TopoDS::Face(aFace));
      BRepBuilderAPI_MakeFace MK1 (aGS, theWire);
      if (MK1.IsDone()) {
        TopoDS_Shape aFace1 = MK1.Shape();
	
        BRepCheck_Analyzer ana (aFace1, false);
        if (!ana.IsValid()) {
          TopoDS_Shape aFace2;
          ShHealOper_ShapeProcess aHealer;
          aHealer.Perform(aFace1, aFace2);
          if (aHealer.isDone())
            theResult = aFace2;
        }
      }
      
      if (theResult.IsNull()) { // try to deal with pure result of filling
        // Update tolerance
        Standard_Real aTol = MF.G0Error();
	
        TColgp_Array1OfPnt aPnts (1,nbEdges); // points of the given wire
        BRepTools_WireExplorer aWE1 (theWire);
        Standard_Integer vi = 1;
        for (; aWE1.More() && vi <= nbEdges; aWE1.Next(), vi++) {
          aPnts(vi) = BRep_Tool::Pnt(TopoDS::Vertex(aWE1.CurrentVertex()));
        }
	
        // Find maximum deviation in vertices
        TopExp_Explorer exp (aFace, TopAbs_VERTEX);
        TopTools_MapOfShape mapShape;
        for (; exp.More(); exp.Next()) {
          if (mapShape.Add(exp.Current())) {
            TopoDS_Vertex aV = TopoDS::Vertex(exp.Current());
            Standard_Real aTolV = BRep_Tool::Tolerance(aV);
            gp_Pnt aP = BRep_Tool::Pnt(aV);
            Standard_Real min_dist = aP.Distance(aPnts(1));
            for (vi = 2; vi <= nbEdges; vi++) {
              min_dist = Min(min_dist, aP.Distance(aPnts(vi)));
            }
            aTol = Max(aTol, aTolV);
            aTol = Max(aTol, min_dist);
          }
        }

	/*
        if ( (*((Handle(BRep_TFace)*)&aFace.TShape()))->Tolerance() < aTol) {
          (*((Handle(BRep_TFace)*)&aFace.TShape()))->Tolerance(aTol);
        }
	//*/

        if ( (Handle(BRep_TFace)::DownCast(aFace.TShape()))->Tolerance() < aTol) {
          (Handle(BRep_TFace)::DownCast(aFace.TShape()))->Tolerance(aTol);
        }


        theResult = aFace;
      }
    }
  } 
  else {
    // try to update wire tolerances to build a planar face
    // Find a deviation
    Standard_Real aToleranceReached, aTol;
    BRepLib_FindSurface aFS;
    aFS.Init(theWire, -1., isPlanarWanted);
    aToleranceReached = aFS.ToleranceReached();
    aTol = aFS.Tolerance();
    
    if (!aFS.Found()) {
      aFS.Init(theWire, aToleranceReached, isPlanarWanted);
      if (!aFS.Found()) return;
      aToleranceReached = aFS.ToleranceReached();
      aTol = aFS.Tolerance();
    }
    aTol = Max(1.2 * aToleranceReached, aTol);

    // Copy the wire, bacause it can be updated with very-very big tolerance here
    BRepBuilderAPI_Copy aMC (theWire);
    if (!aMC.IsDone()) return;
    TopoDS_Wire aWire = TopoDS::Wire(aMC.Shape());
    // Update tolerances to <aTol>
    BRep_Builder B;
    for (TopExp_Explorer expE (aWire, TopAbs_EDGE); expE.More(); expE.Next()) {
      TopoDS_Edge anE = TopoDS::Edge(expE.Current());
      B.UpdateEdge(anE, aTol);
    }
    for (TopExp_Explorer expV (aWire, TopAbs_VERTEX); expV.More(); expV.Next()) {
      TopoDS_Vertex aV = TopoDS::Vertex(expV.Current());
      B.UpdateVertex(aV, aTol);
    }
    //BRepLib::UpdateTolerances(aWire);
    // Build face
    BRepBuilderAPI_MakeFace MK1 (aWire, isPlanarWanted);
    if (MK1.IsDone()) {
      theResult = MK1.Shape();
      return;
    }
  }
}
