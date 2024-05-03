#include <CAGDDefine.hxx>

#include "OCAF_PipeShellDriver.ixx"
#include <OCAF_IPipeShell.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepPrimAPI_MakePrism.hxx>

#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>

#include <BRepNaming_PipeShell.hxx>
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

#define OK_PIPE 0
#define ALGO_NOT_DONE 1
#define EMPTY_PIPE 2

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_PipeShellDriver::OCAF_PipeShellDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_PipeShellDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{

  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_PIPE;
  
  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    if(!aPrevNS->IsEmpty())
      aLocation = aPrevNS->Get().Location();
  }
  
  OCAF_IPipeShell anInterface(aNode);
  Standard_Boolean IsSolid = anInterface.IsSolidOrShell();
  
  TopoDS_Wire aSpine = anInterface.GetSpine();
  TopoDS_Wire aMode = anInterface.GetPipeMode();
  
  BRepOffsetAPI_MakePipeShell aPipeShell(aSpine);

  BRepFill_TypeOfContact tmpContactType = BRepFill_NoContact;
  if(anInterface.IsKeepContact()) tmpContactType = BRepFill_Contact;

  aPipeShell.SetMode(aMode, anInterface.IsCurvilinearEquivalence(), tmpContactType);
  aPipeShell.SetTransitionMode(anInterface.GetTransitionMode());
  

  TopTools_ListOfShape aSections;
  for(int i = 1; i <= aNode->Label().FindChild(ARGUMENTS_TAG).FindChild(PIPESHELL_PROFILE_TAG).NbChildren(); i++) {
    TopoDS_Wire aWire = anInterface.GetProfile(i);
    if(!aWire.IsNull()) {
      aPipeShell.Add(aWire, anInterface.IsWithContact(i), anInterface.IsWithCorrection(i));
      aSections.Append(aWire);
    }
  }

  
  aPipeShell.Build();
  if(IsSolid)
    if(!aPipeShell.MakeSolid())
      return ALGO_NOT_DONE;
  
  if(!aPipeShell.IsDone())
    return ALGO_NOT_DONE;
  
  // Name result
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_PipeShell aNaming(aResultLabel);	
  aNaming.Load(aPipeShell, aSpine, aSections);
  
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);
  
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  
  return OK_PIPE;
}
