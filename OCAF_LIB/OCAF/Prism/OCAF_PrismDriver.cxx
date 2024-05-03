
#include <CAGDDefine.hxx>

#include "OCAF_PrismDriver.ixx"
#include <OCAF_IPrism.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakePrism.hxx>

#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>

#include <BRepNaming_Prism.hxx>
#include <TDF_ChildIterator.hxx>

#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TNaming.hxx>

#include <TopoDS_Shape.hxx>
#include <TopoDS.hxx>
#include <TopAbs.hxx>

#include <gp_Pln.hxx>
#include <Geom_Surface.hxx>
#include <AIS.hxx>
#include <AIS_KindOfSurface.hxx>
#include <gp_Ax1.hxx>
#include <gp_Dir.hxx>
#include <gp_Vec.hxx>
#include <BRepAlgo.hxx>

#include <Standard_NullObject.hxx>


#define OK_PRISM 0
#define PRISM_NOT_DONE 1
#define EMPTY_PRISM 2
#define NULL_PRISM 3
//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_PrismDriver::OCAF_PrismDriver():OCAF_Driver() {}


//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_PrismDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) 
    return EMPTY_PRISM;
  
  OCAF_IPrism anInterface(aNode);
  gp_Vec aVector;
  
  TopoDS_Edge aSpine = anInterface.GetVector();
  TopoDS_Shape aProfile = anInterface.GetProfile();
  
  if (aProfile.IsNull() || aSpine.IsNull() ) {
    Standard_NullObject::Raise("MakePrism aborted : null base or spine argument");
  }
  
  
  TopoDS_Vertex V1, V2;
  TopExp::Vertices(aSpine, V1, V2, Standard_True);
  if (V1.IsNull() || V2.IsNull()) {
    Standard_NullObject::Raise("Prism creation aborted: vector is not defined");
  }
  aVector = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
  
  BRepPrimAPI_MakePrism mkPrism(aProfile, aVector);
  
  mkPrism.Build();
  
  if(!mkPrism.IsDone()) return PRISM_NOT_DONE;
  
  TopoDS_Shape aShape = mkPrism.Shape();
  
  if(aShape.IsNull()) return NULL_PRISM;
  
  // Name result
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Prism aNaming(aResultLabel);	
  aNaming.Load(aShape, BRepNaming_PRISM);
  
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);
  
  return OK_PRISM;
}

