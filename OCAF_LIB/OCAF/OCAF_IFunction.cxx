#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>
#include <OCAF_ITranslate.hxx>

#include <OCAF_IDisplayer.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Label.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_ChildNodeIterator.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_Name.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_UAttribute.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Scope.hxx>

#include <TNaming_NamedShape.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_RefShape.hxx>
#include <TNaming_UsedShapes.hxx>
#include <TNaming_PtrNode.hxx>

#include <TPrsStd_AISPresentation.hxx>

#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TCollection_AsciiString.hxx>

#include <Standard_ConstructionError.hxx>


//--------------------------------------------->>>
#include <OCAF_IVertex.hxx>
#include <OCAF_IEdge.hxx>
#include <OCAF_IWire.hxx>
#include <OCAF_IFace.hxx>
#include <OCAF_IShell.hxx>
#include <OCAF_ISolid.hxx>

#include <OCAF_IThruSections.hxx>
#include <OCAF_IPipeShell.hxx>
#include <OCAF_IPipe.hxx>


#include <OCAF_IBox.hxx>
#include <OCAF_ICylinder.hxx>
#include <OCAF_ISphere.hxx>
#include <OCAF_ITorus.hxx>
#include <OCAF_ICone.hxx>

#include <OCAF_IPrism.hxx>
#include <OCAF_IRevolution.hxx>


#include <OCAF_ICircle.hxx>
#include <OCAF_IEllipse.hxx>
#include <OCAF_IParabola.hxx>
#include <OCAF_IArc.hxx>
#include <OCAF_IVector.hxx>
#include <OCAF_IPolygon.hxx>

#include <OCAF_ICut.hxx>
#include <OCAF_IMultiCut.hxx>
#include <OCAF_ICommon.hxx>
#include <OCAF_IFuse.hxx>
#include <OCAF_IMultiFuse.hxx>

#include <OCAF_ITranslate.hxx>
#include <OCAF_IMirror.hxx>
#include <OCAF_IRotate.hxx>
#include <OCAF_IMultiRotate.hxx>
#include <OCAF_IPeriodShape.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_IBRepImport.hxx>


//#include <OCAF_IFillet.hxx>


#include <OCAF_ICosPeriodEdge.hxx>
#include <OCAF_IRecPeriodEdge.hxx>
#include <OCAF_IHelixEdge.hxx>

#include <OCAF_ICurve.hxx>

#include <OCAF_ObjectTool.hxx>


OCAF_ObjectType 
OCAF_IFunction::
GetObjectType(const Handle(TDataStd_TreeNode)& Object)
{
  Handle(TFunction_Function) aFunction;
  if (!Object->FindAttribute(TFunction_Function::GetID(), aFunction))
    return OCAF_NotDefinedObjectType;
  
  const Standard_GUID DriverGUID =  aFunction->GetDriverGUID();

  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IBox::GetID()))         return OCAF_Box;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICylinder::GetID()))    return OCAF_Cylinder;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ISphere::GetID()))      return OCAF_Sphere;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICone::GetID()))        return OCAF_Cone;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ITorus::GetID()))       return OCAF_Torus;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IPrism::GetID()))       return OCAF_Prism;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IRevolution::GetID()))  return OCAF_Revolution;
  /****************************************************************/


  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IPolygon::GetID()))   return OCAF_Polygon;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICircle::GetID()))    return OCAF_Circle;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IEllipse::GetID()))   return OCAF_Ellipse;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IParabola::GetID()))  return OCAF_Parabola;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IArc::GetID()))       return OCAF_Arc;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IVector::GetID()))    return OCAF_Vector;
  /****************************************************************/


  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IVertex::GetID()))   return OCAF_Vertex;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IEdge::GetID()))     return OCAF_Edge;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IWire::GetID()))     return OCAF_Wire;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IFace::GetID()))     return OCAF_Face;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IShell::GetID()))    return OCAF_Shell;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ISolid::GetID()))    return OCAF_Solid;
  /****************************************************************/


  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICut::GetID()))       return OCAF_Cut;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IMultiCut::GetID()))  return OCAF_MultiCut;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IFuse::GetID()))      return OCAF_Fuse;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IMultiFuse::GetID()))      return OCAF_MultiFuse;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICommon::GetID()))    return OCAF_Common;
  /****************************************************************/


  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ITranslate::GetID()))  return OCAF_Translate;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IMirror::GetID()))     return OCAF_Mirror;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IRotate::GetID()))     return OCAF_Rotate;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IPeriodShape::GetID()))return OCAF_PeriodShape;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IMultiRotate::GetID()))return OCAF_MultiRotate;
  /****************************************************************/


  /****************************************************************/
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IThruSections::GetID()))  return OCAF_ThruSections;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IPipeShell::GetID()))     return OCAF_PipeShell;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IPipe::GetID()))          return OCAF_Pipe;
  /****************************************************************/

  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IBRepImport::GetID()))    return OCAF_BRepImport;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ISelection::GetID()))     return OCAF_Selection;


  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICosPeriodEdge::GetID()))     return OCAF_CosPeriodEdge;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IRecPeriodEdge::GetID()))     return OCAF_RecPeriodEdge;
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IHelixEdge::GetID()))         return OCAF_HelixEdge;
  /*
  if (Standard_GUID::IsEqual(DriverGUID, OCAF_IFillet::GetID()))    return OCAF_Fillet;
  */


  if (Standard_GUID::IsEqual(DriverGUID, OCAF_ICurve::GetID()))  return OCAF_Curve;
  return OCAF_NotDefinedObjectType;
}



//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_IFunction::OCAF_IFunction(const Handle(TDataStd_TreeNode)& aTreeNode):myTreeNode(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode");
}



//=============================================================================
//function : GetEntry
//purpose  :
//=============================================================================
TDF_Label 
OCAF_IFunction::
GetEntry() const
{
  if (myTreeNode.IsNull()) return TDF_Label();
  return myTreeNode->Label();
}



//=======================================================================
//function : SetName
//purpose  :
//=======================================================================
void OCAF_IFunction::SetName(const TCollection_ExtendedString& aName)
{
  if (myTreeNode.IsNull()) return;

  TDataStd_Name::Set(myTreeNode->Label(), aName);
  TDocStd_Modified::Add(myTreeNode->Label());
}


//=======================================================================
//function : HasName
//purpose  :
//=======================================================================

Standard_Boolean OCAF_IFunction::HasName()
{
  if (myTreeNode.IsNull()) return Standard_False;

  Handle(TDataStd_Name) aName;
  return myTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName);
}


//=======================================================================
//function : GetName
//purpose  :
//=======================================================================

TCollection_ExtendedString OCAF_IFunction::GetName() const
{
  TCollection_ExtendedString anExtendedString;
  if (myTreeNode.IsNull()) return anExtendedString;

  Handle(TDataStd_Name) aName;
  if (!myTreeNode->Label().FindAttribute(TDataStd_Name::GetID(),aName)) return anExtendedString;

  anExtendedString = aName->Get();
  return anExtendedString;
}



//=======================================================================
//function : AddLabels
//purpose  : Adds all result labels of the object to theLog    ??????????????????????????????????????
//=======================================================================
void OCAF_IFunction::AddLabels(const Handle(TDataStd_TreeNode)& theFunctionNode, Handle(TFunction_Logbook)& theLog)
{
  if(theFunctionNode.IsNull()) return;

  TDF_ChildIterator anIterator;
  Handle(TDataStd_TreeNode) aNode = theFunctionNode->Previous();        // Origin ---marked 2015.12.09
  //Handle(TDataStd_TreeNode) aNode = theFunctionNode;                  // Modified by Wang Yue 2010.04.01

  while(!aNode.IsNull()) {
    anIterator.Initialize(aNode->Label().FindChild(RESULTS_TAG));
    for(; anIterator.More(); anIterator.Next()) {
      if(anIterator.Value().IsAttribute(TNaming_NamedShape::GetID())) {
	theLog->SetImpacted(anIterator.Value());
      }
    }
    aNode = aNode->Previous();
  }
}



void OCAF_IFunction::AddLogBooks(const Handle(TDataStd_TreeNode)& theFunctionNode, Handle(TFunction_Logbook)& theLog)
{
  if(OCAF_ObjectTool::IsOneFunctionNode(theFunctionNode)){
    //theLog = TFunction_Logbook::Set(theFunctionNode->Label());
    theLog = TFunction_Logbook::Set(theFunctionNode->Root()->Label());

    TDF_Label aResultLabel = theFunctionNode->Label().FindChild(RESULTS_TAG);

    // 1   to mark "aNode" father's label as modified 
    TDocStd_Modified::Add(theFunctionNode->Father()->Label());

    AddLabels(theFunctionNode, theLog);
    
    // 2   to mark "aNode" as modified
    theLog->SetImpacted(theFunctionNode->Label());

    TDocStd_Modified::Add(theFunctionNode->Label()); 
    
    // 3   to mark "aResultLabel" as modified
    theLog->SetImpacted(aResultLabel);

    // 4   to mark "aResultLabel" children as Impacted!
    TDF_ChildIterator anIterator(aResultLabel);
    for(; anIterator.More(); anIterator.Next()) {
      theLog->SetImpacted(anIterator.Value());
    }
  }
}


//=======================================================================
//function : GetFunctionResult
//purpose  :
//=======================================================================
TopoDS_Shape OCAF_IFunction::GetFunctionResult()
{
  if(myTreeNode.IsNull()) return TopoDS_Shape();
  
  Handle(TNaming_NamedShape) aNS;
  TDF_Label aLabel = myTreeNode->Label().FindChild(RESULTS_TAG);
  if(!aLabel.IsNull() && aLabel.FindAttribute(TNaming_NamedShape::GetID(), aNS)) {
    return aNS->Get();
  }
  return TopoDS_Shape();
}

