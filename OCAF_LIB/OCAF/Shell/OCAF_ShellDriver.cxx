// File:	OCAF_PointDriver.cxx
// Created:	Thu Feb  7 15:41:07 2002
// Author:	Michael KUZMITCHEV
//		<mkv@russox.nnov.matra-dtv.fr>
//Modified by:  Sergey RUIN (Naming)
#include <CAGDDefine.hxx>


#include "OCAF_ShellDriver.ixx"
#include <OCAF_IShell.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_MakeShell.hxx>

#include <BRepNaming_Shell.hxx>
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
#include <TopExp_Explorer.hxx>


#define OK_SHELL 0
#define EMPTY_SHELL 1
#define SHELL_NOT_DONE 2
#define NULL_SHELL 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ShellDriver::OCAF_ShellDriver():OCAF_Driver() {}


//*
//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ShellDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_SHELL", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SHELL;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IShell "anInterface"
  OCAF_IShell anInterface(aNode);

  BRepBuilderAPI_Sewing aSewing(Precision::Confusion()*10.0);

  TDF_AttributeMap aRefmap;
  anInterface.GetBuildShellElementsMap(aRefmap);
  Standard_Integer ind;
  Standard_Integer nbelements = aRefmap.Extent();

  for(ind = 1; ind<= nbelements; ind++){
    TopoDS_Shape aElement_i = anInterface.GetBuildShellElement(ind);
    if (aElement_i.IsNull()) {
      Standard_NullObject::Raise("Shape for wire construction is null");
    }
    if( aElement_i.ShapeType() == TopAbs_FACE ) {
      aSewing.Add( TopoDS::Face(aElement_i) );
    }
    else if( aElement_i.ShapeType() == TopAbs_SHELL ) {
      TopExp_Explorer anExp (aElement_i, TopAbs_FACE);
      for (; anExp.More(); anExp.Next()) {
          aSewing.Add(anExp.Current());
      }
      //aSewing.Add( TopoDS::Shell(aElement_i)); //  will it work?  2010.04.22
    }
    else{
      Standard_TypeMismatch::Raise("Shape for wire construction is neither an edge nor a wire");
    }
  }
  aSewing.Perform();


  TopoDS_Shape aSewedShape = aSewing.SewedShape();
  TopoDS_Shell aShell;


  if( aSewedShape.ShapeType() == TopAbs_FACE && nbelements == 1 ) {
    // case for creation of shell from one face
    BRep_Builder B;
    B.MakeShell(aShell);
    B.Add(aShell, aSewedShape);
  }
  else {
    TopExp_Explorer exp (aSewedShape, TopAbs_SHELL);
    Standard_Integer iface = 0;
    for (; exp.More(); exp.Next()) {
      aShell = TopoDS::Shell(exp.Current());
      iface++;
    }
    if (iface != 1)
      aShell = TopoDS::Shell(aSewedShape);
  }
  

  // 5. make a wire using the BRepNaming_Shell method.
  if (aShell.IsNull()) return NULL_SHELL;
  if ( !BRepAlgo::IsValid(aShell) ) return SHELL_NOT_DONE;    //???????????????

  /*********************************************************************/
  Standard_Integer anOrientation = anInterface.GetOrientation();
  if(anOrientation == 1) {
    aShell.Orientation(TopAbs_FORWARD);
  }
  else if(anOrientation == 2){
    aShell.Orientation(TopAbs_REVERSED);
  }
  /********************************************************************/

  // 6. create a child label of this driver's label
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Shell aNaming(aResultLabel);
  aNaming.Load(aShell, BRepNaming_SHELL);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_SHELL;
}

//*/








/*
//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ShellDriver::Execute(TFunction_Logbook& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_SHELL", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_SHELL;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IShell "anInterface"
  OCAF_IShell anInterface(aNode);

  TDF_Label aResultLabel;


  BRep_Builder B;
  TopoDS_Shell aShell;
  B.MakeShell(aShell);


  TDF_AttributeMap aRefmap;
  anInterface.GetBuildShellElementsMap(aRefmap);
  Standard_Integer ind;
  Standard_Integer nbelements = aRefmap.Extent();

  for(ind = 0; ind< nbelements; ind++){
    TopoDS_Shape aElement_i = anInterface.GetBuildShellElement(ind);
    if (aElement_i.IsNull()) {
      Standard_NullObject::Raise("Shape for wire construction is null");
    }
    if( aElement_i.ShapeType() == TopAbs_FACE ) {
      B.Add(aShell, TopoDS::Face(aElement_i));
    }
    else if( aElement_i.ShapeType() == TopAbs_SHELL ) {
      //B.Add(aShell, TopoDS::Shell(aElement_i));
      TopExp_Explorer anExp (aElement_i, TopAbs_FACE);
      for (; anExp.More(); anExp.Next()) {
	B.Add(aShell, TopoDS::Shell(anExp.Current()));
      }
    }
    else{
      Standard_TypeMismatch::Raise("Shape for wire construction is neither an edge nor a wire");
    }
  }

  // 5. make a wire using the BRepNaming_Shell method.
  if (aShell.IsNull()) return NULL_SHELL;
  if ( !BRepAlgo::IsValid(aShell) ) return SHELL_NOT_DONE;
  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Shell aNaming(aResultLabel);
  aNaming.Load(aShell, BRepNaming_SHELL);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_SHELL;
}

//*/
