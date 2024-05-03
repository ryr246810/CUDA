// File:	OCAF_PointDriver.cxx
// Created:	Thu Feb  7 15:41:07 2002
// Author:	Michael KUZMITCHEV
//		<mkv@russox.nnov.matra-dtv.fr>
//Modified by:  Sergey RUIN (Naming)
#include <CAGDDefine.hxx>


#include "OCAF_SolidDriver.ixx"
#include <OCAF_ISolid.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeSolid.hxx>

#include <BRepNaming_Solid.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#include <BRepClass3d_SolidClassifier.hxx>

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
#include <TopExp_Explorer.hxx>


#define OK_SOLID 0
#define EMPTY_SOLID 1
#define SOLID_NOT_DONE 2
#define NULL_SOLID 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_SolidDriver::OCAF_SolidDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_SolidDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_SOLID", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SOLID;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_ISolid "anInterface"
  OCAF_ISolid anInterface(aNode);

  TDF_Label aResultLabel;


  BRep_Builder B;
  TopoDS_Solid aSolid;
  B.MakeSolid(aSolid);


  TDF_AttributeMap aRefmap;
  anInterface.GetBuildSolidElementsMap(aRefmap);
  Standard_Integer ind;
  Standard_Integer nbelements = aRefmap.Extent();

  for(ind = 1; ind<= nbelements; ind++){
    TopoDS_Shape aElement_i = anInterface.GetBuildSolidElement(ind);
    if (aElement_i.IsNull()) {
      Standard_NullObject::Raise("Shape for wire construction is null");
    }
    if( aElement_i.ShapeType() == TopAbs_SHELL ) {
      B.Add(aSolid, TopoDS::Shell(aElement_i));
    }
    else{
      Standard_TypeMismatch::Raise("Shape for wire construction is neither an edge nor a wire");
    }
  }

  //*
  BRepClass3d_SolidClassifier SC (aSolid);
  SC.PerformInfinitePoint(Precision::Confusion());
  if (SC.State() == TopAbs_IN)
    aSolid.Reverse();
  //*/

  // 5. make a wire using the BRepNaming_Solid method.
  if (aSolid.IsNull())  return NULL_SOLID;
  if ( !BRepAlgo::IsValid(aSolid) )  return SOLID_NOT_DONE;

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Solid aNaming(aResultLabel);
  aNaming.Load(aSolid, BRepNaming_SOLID);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_SOLID;
}

