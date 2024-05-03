#include <CAGDDefine.hxx>

#include "OCAF_PipeDriver.ixx"
#include <OCAF_IPipe.hxx>
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

#include <BRepNaming_Pipe.hxx>
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


#include <TopExp_Explorer.hxx> 
#include <GeomAdaptor_Curve.hxx>
#include <GeomAdaptor_HCurve.hxx>
#include <GeomFill_CorrectedFrenet.hxx>


#include <Standard_NullObject.hxx>


#define OK_PIPE 0
#define PIPE_NOT_DONE 1
#define EMPTY_PIPE 2
#define NULL_PIPE 3
//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_PipeDriver::OCAF_PipeDriver():OCAF_Driver() {}



//=======================================================================
//function : EvaluateBestSweepMode
//purpose  : auxilary for right call of MakePipe and MakePipeShell
//=======================================================================
static GeomFill_Trihedron EvaluateBestSweepMode(const TopoDS_Shape& Spine)
{
  GeomFill_Trihedron theMode = GeomFill_IsFrenet;
  
  TopExp_Explorer Explo(Spine, TopAbs_EDGE);
  for (; Explo.More(); Explo.Next())
  {
    TopoDS_Edge anEdge = TopoDS::Edge(Explo.Current());
    Standard_Real fpar, lpar;
    Handle(Geom_Curve) aCurve = BRep_Tool::Curve(anEdge, fpar, lpar);
    GeomAdaptor_Curve GAcurve(aCurve, fpar, lpar);
    Handle(GeomAdaptor_HCurve) GAHcurve = new GeomAdaptor_HCurve(GAcurve);

    Handle(GeomFill_CorrectedFrenet) aCorrFrenet = new GeomFill_CorrectedFrenet(Standard_True); //for evaluation
    aCorrFrenet->SetCurve(GAHcurve);
    GeomFill_Trihedron aMode = aCorrFrenet->EvaluateBestMode();
    if (aMode == GeomFill_IsDiscreteTrihedron)
    {
      theMode = aMode;
      break;
    }
    if (aMode == GeomFill_IsCorrectedFrenet)
      theMode = aMode;
  }
  return theMode;
}



//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_PipeDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_PIPE;

  OCAF_IPipe anInterface(aNode);
  
  TopoDS_Wire aSpine = anInterface.GetSpine();
  TopoDS_Shape aProfile = anInterface.GetProfile();
  
  if (aProfile.IsNull() || aSpine.IsNull() ) {
    Standard_NullObject::Raise("MakePipe aborted : null base or spine argument");
  }

  GeomFill_Trihedron theBestMode = EvaluateBestSweepMode(aSpine);
  BRepOffsetAPI_MakePipe mkPipe(aSpine, aProfile, theBestMode, Standard_False);
  
  mkPipe.Build();
  if(!mkPipe.IsDone()){
    return PIPE_NOT_DONE;
  }
  TopoDS_Shape aShape = mkPipe.Shape();

  if(aShape.IsNull()){
    return NULL_PIPE;
  }
  // Name result
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_Pipe aNaming(aResultLabel);	
  aNaming.Load(aShape, BRepNaming_PIPE);

  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_PIPE;
}
