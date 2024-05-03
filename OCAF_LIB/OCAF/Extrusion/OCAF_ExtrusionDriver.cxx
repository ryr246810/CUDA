#include <CAGDDefine.hxx>

#include "OCAF_ExtrusionDriver.ixx"
#include <OCAF_IExtrusion.hxx>
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

#include <BRepNaming_Extrusion.hxx>
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


#define OK_EXTRUSION 0
#define EMPTY_EXTRUSION 1
#define EXTRUSION_NOT_DONE 2
#define NULL_EXTRUSION 3
//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_ExtrusionDriver::OCAF_ExtrusionDriver():OCAF_Driver() {}


//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ExtrusionDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) 
    return EMPTY_EXTRUSION;
  
  OCAF_IExtrusion anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;
  
  TopoDS_Shape aShape;
  
  gp_Vec aVector;
  
  if (aType == EXTRUSION_BASE_VEC_H) {
    TopoDS_Edge aSpine = anInterface.GetVector();
    TopoDS_Shape aProfile = anInterface.GetProfile();
    
    if (aProfile.IsNull() || aSpine.IsNull() ) {
      Standard_NullObject::Raise("MakeExtrusion aborted : null base or spine argument");
    }
    
    TopoDS_Vertex V1, V2;
    TopExp::Vertices(aSpine, V1, V2, Standard_True);
    if (V1.IsNull() || V2.IsNull()) {
      Standard_NullObject::Raise("Extrusion creation aborted: vector is not defined");
    }
    aVector = gp_Vec(BRep_Tool::Pnt(V1), BRep_Tool::Pnt(V2));
    
    BRepPrimAPI_MakePrism mkExtrusion(aProfile, aVector);
    mkExtrusion.Build();
    
    if(!mkExtrusion.IsDone())  return EXTRUSION_NOT_DONE;

    aShape = mkExtrusion.Shape();
  }else if (aType == EXTRUSION_BASE_TWO_PNT) {
  }else if (aType == EXTRUSION_BASE_DX_DY_DZ){ 
  }else{
  }
  
  
  
  //////////////////////////////////////////////
  if(aShape.IsNull()) return NULL_EXTRUSION;
  
  // Name result
  aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Extrusion aNaming(aResultLabel);	
  aNaming.Load(aShape, BRepNaming_EXTRUSION);
  
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);
  
  return OK_EXTRUSION;
}
