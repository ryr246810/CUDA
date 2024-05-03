
#include <CAGDDefine.hxx>


#include "OCAF_BoxDriver.ixx"
#include <OCAF_IBox.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakeBox.hxx>

#include <BRepNaming_Box.hxx>
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

#include <StdFail_NotDone.hxx>

#include <BRepAlgo.hxx>



#define OK_BOX 0
#define X_NOT_FOUND 1
#define Y_NOT_FOUND 2
#define Z_NOT_FOUND 3
#define DX_NOT_FOUND 4
#define DY_NOT_FOUND 5
#define DZ_NOT_FOUND 6
#define EMPTY_BOX 7
#define BOX_NOT_DONE 8
#define NULL_BOX 9

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_BoxDriver::OCAF_BoxDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_BoxDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_BOX", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_BOX;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;


  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_IBox anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();

  TDF_Label aResultLabel;

  if (aType == BOX_DIMENSIONS) {

    if (anInterface.GetDX() == 0 && 
	anInterface.GetDY() == 0 && 
	anInterface.GetDZ() == 0) return EMPTY_BOX;

    // 5_1. make a box using the BRepNaming_Box method.
    BRepPrimAPI_MakeBox mkBox( gp_Pnt(anInterface.GetX(), anInterface.GetY(), anInterface.GetZ()), 
			       anInterface.GetDX(), anInterface.GetDY(), anInterface.GetDZ() );
    mkBox.Build();

    if (!mkBox.IsDone()) return BOX_NOT_DONE;
    if (mkBox.Shape().IsNull()) return NULL_BOX;
    if (!BRepAlgo::IsValid(mkBox.Shape())) return BOX_NOT_DONE;

    // 6_1. create a child label of this driver's label
    aResultLabel = Label().FindChild(RESULTS_TAG);
    // 7_1. append a TNaming_NamedShape attribute to "aResultLabel"
    BRepNaming_Box aNaming(aResultLabel);
    aNaming.Load(mkBox, BRepNaming_SOLID);
  }
  else if(aType == BOX_TWO_PNT){
    TopoDS_Shape aShape1 = anInterface.GetPoint1();
    TopoDS_Shape aShape2 = anInterface.GetPoint2();

    if (aShape1.ShapeType() == TopAbs_VERTEX && aShape2.ShapeType() == TopAbs_VERTEX) {
      gp_Pnt P1 = BRep_Tool::Pnt(TopoDS::Vertex(aShape1));
      gp_Pnt P2 = BRep_Tool::Pnt(TopoDS::Vertex(aShape2));

      if (P1.X() == P2.X() || P1.Y() == P2.Y() || P1.Z() == P2.Z()) {
	StdFail_NotDone::Raise("Box can not be created, the points belong to the same plane");
	return EMPTY_BOX; //return 0;
      }

      // 5_2. make a box using the BRepNaming_Box method.
      BRepPrimAPI_MakeBox  mkBox(P1,P2);
      mkBox.Build();

      if (!mkBox.IsDone()) return BOX_NOT_DONE;
      if (mkBox.Shape().IsNull()) return NULL_BOX;
      if (!BRepAlgo::IsValid(mkBox.Shape())) return BOX_NOT_DONE;

      // 6_2. create a child label of this driver's label
      aResultLabel = Label().FindChild(RESULTS_TAG);
      // 7_2. append a TNaming_NamedShape attribute to "aResultLabel"
      BRepNaming_Box aNaming(aResultLabel);
      aNaming.Load(mkBox, BRepNaming_SOLID);
    }
  }
  else{
    return EMPTY_BOX;
  }


  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);


  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_BOX;
}

