
#include <CAGDDefine.hxx>

#include "OCAF_SphereDriver.ixx"
#include <OCAF_ISphere.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakeSphere.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>

#include <BRepNaming_Sphere.hxx>
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#include <TDF_ChildIterator.hxx>

#include <TopoDS_Shape.hxx>
#include <gp_Pnt.hxx>
#include <BRepAlgo.hxx>

#define OK_SPHERE 0
#define X_NOT_FOUND 1
#define Y_NOT_FOUND 2
#define Z_NOT_FOUND 3
#define RADIUS_NOT_FOUND 4
#define EMPTY_SPHERE 5
#define SPHERE_NOT_DONE 6
#define NULL_SPHERE 7

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_SphereDriver::OCAF_SphereDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_SphereDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const {
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SPHERE;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    if(!aPrevNS->IsEmpty())
      aLocation = aPrevNS->Get().Location();
  }
  
  OCAF_ISphere anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;
  TopoDS_Shape aShape;
  
  if (aType == SPHERE_R) {
    double anR = anInterface.GetR();
    if (anR < Precision::Confusion())
      Standard_ConstructionError::Raise("Sphere creation aborted: radius value less than 1e-07 is not acceptable");
    aShape = BRepPrimAPI_MakeSphere(anR).Shape();
  }
  else if (aType == SPHERE_PNT_R) {
    double anR = anInterface.GetR();
    if (anR < Precision::Confusion())
      Standard_ConstructionError::Raise("Sphere creation aborted: radius value less than 1e-07 is not acceptable");

    TopoDS_Shape aRefPoint  = anInterface.GetPoint();
    if (aRefPoint.ShapeType() != TopAbs_VERTEX)
      Standard_ConstructionError::Raise("Invalid shape given for sphere center: it must be a point");
    gp_Pnt aP = BRep_Tool::Pnt(TopoDS::Vertex(aRefPoint));

    aShape = BRepPrimAPI_MakeSphere(aP, anR).Shape();
  }
  else {
  }

  if (aShape.IsNull()) return NULL_SPHERE;
  // Name result
  aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Sphere aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_SOLID);

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_SPHERE;
}
