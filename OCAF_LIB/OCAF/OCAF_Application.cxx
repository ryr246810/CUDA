#include <TFunction_DriverTable.hxx>
#include <TPrsStd_DriverTable.hxx>

#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TDF_Label.hxx>
#include <TDF_Data.hxx>
#include <TDataStd_TreeNode.hxx>

/**********************************/
#include <OCAF_AISFunctionDriver.hxx>

#include <OCAF_ISelection.hxx>
#include <OCAF_SelectionDriver.hxx>

#include <OCAF_IBRepImport.hxx>
#include <OCAF_BRepImportDriver.hxx>

#include <OCAF_ISTEPImport.hxx>
#include <OCAF_STEPImportDriver.hxx>
/**********************************/


/**********************************/
#include <OCAF_IBox.hxx>
#include <OCAF_BoxDriver.hxx>

#include <OCAF_ICylinder.hxx>
#include <OCAF_CylinderDriver.hxx>

#include <OCAF_ISphere.hxx>
#include <OCAF_SphereDriver.hxx>

#include <OCAF_ICone.hxx>
#include <OCAF_ConeDriver.hxx>

#include <OCAF_ITorus.hxx>
#include <OCAF_TorusDriver.hxx>
/**********************************/


/**********************************/
#include <OCAF_IVertex.hxx>
#include <OCAF_VertexDriver.hxx>

#include <OCAF_IEdge.hxx>
#include <OCAF_EdgeDriver.hxx>

#include <OCAF_IWire.hxx>
#include <OCAF_WireDriver.hxx>

#include <OCAF_IFace.hxx>
#include <OCAF_FaceDriver.hxx>

#include <OCAF_IShell.hxx>
#include <OCAF_ShellDriver.hxx>

#include <OCAF_ISolid.hxx>
#include <OCAF_SolidDriver.hxx>
/**********************************/


/**********************************/
#include <OCAF_IThruSections.hxx>
#include <OCAF_ThruSectionsDriver.hxx>

#include <OCAF_IPipeShell.hxx>
#include <OCAF_PipeShellDriver.hxx>

#include <OCAF_IPipe.hxx>
#include <OCAF_PipeDriver.hxx>

#include <OCAF_IRevolution.hxx>
#include <OCAF_RevolutionDriver.hxx>
/**********************************/


/**********************************/
#include <OCAF_IPolygon.hxx>
#include <OCAF_PolygonDriver.hxx>

#include <OCAF_ICircle.hxx>
#include <OCAF_CircleDriver.hxx>

#include <OCAF_IArc.hxx>
#include <OCAF_ArcDriver.hxx>

#include <OCAF_IEllipse.hxx>
#include <OCAF_EllipseDriver.hxx>

#include <OCAF_IParabola.hxx>
#include <OCAF_ParabolaDriver.hxx>

#include <OCAF_IVector.hxx>
#include <OCAF_VectorDriver.hxx>
/**********************************/


/**********************************/
#include <OCAF_ICut.hxx>
#include <OCAF_CutDriver.hxx>

#include <OCAF_ICommon.hxx>
#include <OCAF_CommonDriver.hxx>

#include <OCAF_IFuse.hxx>
#include <OCAF_FuseDriver.hxx>

#include <OCAF_IMultiFuse.hxx>
#include <OCAF_MultiFuseDriver.hxx>

#include <OCAF_IMultiCut.hxx>
#include <OCAF_MultiCutDriver.hxx>
/**********************************/

/**********************************/
#include <OCAF_ITranslate.hxx>
#include <OCAF_TranslateDriver.hxx>

#include <OCAF_IPeriodShape.hxx>
#include <OCAF_PeriodShapeDriver.hxx>

#include <OCAF_IMirror.hxx>
#include <OCAF_MirrorDriver.hxx>

#include <OCAF_IRotate.hxx>
#include <OCAF_RotateDriver.hxx>

#include <OCAF_IMultiRotate.hxx>
#include <OCAF_MultiRotateDriver.hxx>
/**********************************/


#include <OCAF_IPrism.hxx>
#include <OCAF_PrismDriver.hxx>


#include <OCAF_ICosPeriodEdge.hxx>
#include <OCAF_CosPeriodEdgeDriver.hxx>


#include <OCAF_IRecPeriodEdge.hxx>
#include <OCAF_RecPeriodEdgeDriver.hxx>


#include <OCAF_IHelixEdge.hxx>
#include <OCAF_HelixEdgeDriver.hxx>



#include <OCAF_ICurve.hxx>
#include <OCAF_CurveDriver.hxx>
/*
#include <OCAF_IFillet.hxx>
#include <OCAF_FilletDriver.hxx>
*/

#include "OCAF_Application.ixx"

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_Application::OCAF_Application()
{
  //Add	all drivers to the static tables

  TFunction_DriverTable::Get()->AddDriver(OCAF_IBox::GetID(), new OCAF_BoxDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ICylinder::GetID(), new OCAF_CylinderDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ISphere::GetID(), new OCAF_SphereDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ICone::GetID(), new OCAF_ConeDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ITorus::GetID(), new OCAF_TorusDriver());

  TFunction_DriverTable::Get()->AddDriver(OCAF_IVertex::GetID(), new OCAF_VertexDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IEdge::GetID(), new OCAF_EdgeDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IWire::GetID(), new OCAF_WireDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IFace::GetID(), new OCAF_FaceDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IShell::GetID(), new OCAF_ShellDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ISolid::GetID(), new OCAF_SolidDriver());

  TFunction_DriverTable::Get()->AddDriver(OCAF_IPolygon::GetID(), new OCAF_PolygonDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ICircle::GetID(), new OCAF_CircleDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IEllipse::GetID(), new OCAF_EllipseDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IParabola::GetID(), new OCAF_ParabolaDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IArc::GetID(), new OCAF_ArcDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IVector::GetID(), new OCAF_VectorDriver());


  TFunction_DriverTable::Get()->AddDriver(OCAF_ICut::GetID(), new OCAF_CutDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IMultiCut::GetID(), new OCAF_MultiCutDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IFuse::GetID(), new OCAF_FuseDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IMultiFuse::GetID(), new OCAF_MultiFuseDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ICommon::GetID(), new OCAF_CommonDriver());

  TFunction_DriverTable::Get()->AddDriver(OCAF_ITranslate::GetID(), new OCAF_TranslateDriver()); 
  TFunction_DriverTable::Get()->AddDriver(OCAF_IMirror::GetID(), new OCAF_MirrorDriver()); 
  TFunction_DriverTable::Get()->AddDriver(OCAF_IRotate::GetID(), new OCAF_RotateDriver()); 
  TFunction_DriverTable::Get()->AddDriver(OCAF_IPeriodShape::GetID(), new OCAF_PeriodShapeDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IMultiRotate::GetID(), new OCAF_MultiRotateDriver()); 


  TFunction_DriverTable::Get()->AddDriver(OCAF_IRevolution::GetID(), new OCAF_RevolutionDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IThruSections::GetID(), new OCAF_ThruSectionsDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IPipeShell::GetID(), new OCAF_PipeShellDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IPipe::GetID(), new OCAF_PipeDriver());

  TFunction_DriverTable::Get()->AddDriver(OCAF_IBRepImport::GetID(), new OCAF_BRepImportDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_ISTEPImport::GetID(), new OCAF_STEPImportDriver());

  TFunction_DriverTable::Get()->AddDriver(OCAF_ISelection::GetID(), new OCAF_SelectionDriver()); 

  TFunction_DriverTable::Get()->AddDriver(OCAF_IPrism::GetID(), new OCAF_PrismDriver());
  //TFunction_DriverTable::Get()->AddDriver(OCAF_IFillet::GetID(), new OCAF_FilletDriver());   

  TFunction_DriverTable::Get()->AddDriver(OCAF_ICosPeriodEdge::GetID(), new OCAF_CosPeriodEdgeDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IRecPeriodEdge::GetID(), new OCAF_RecPeriodEdgeDriver());
  TFunction_DriverTable::Get()->AddDriver(OCAF_IHelixEdge::GetID(), new OCAF_HelixEdgeDriver());


  TFunction_DriverTable::Get()->AddDriver(OCAF_ICurve::GetID(), new OCAF_CurveDriver());


  TPrsStd_DriverTable::Get()->AddDriver(OCAF_AISFunctionDriver::GetID(), new OCAF_AISFunctionDriver());
}

//=======================================================================
//function : InitDocument
//purpose  : Called after creation of the document
//=======================================================================
void OCAF_Application::InitDocument(const Handle(TDocStd_Document)& theDoc) const
{
  TDF_Label aRootLabel = theDoc->GetData()->Root();
  //Set a TreeNode to the root label as the root TreeNode of the document
  Handle(TDataStd_TreeNode) aRootTreeNode = TDataStd_TreeNode::Set(aRootLabel);

  TNaming_Builder aBuilder(theDoc->Main());
  theDoc->Main().ForgetAttribute(TNaming_NamedShape::GetID());
  //Protected against a deletion of UsedShapes attribute on Undo (work around, till the bug is fixed)
}

//=======================================================================
//function : ResourcesName
//purpose  : Name of the resource file 
//=======================================================================
Standard_CString OCAF_Application::ResourcesName()
{
  const Standard_CString aRes = "Standard";
  return aRes;
}

//=======================================================================
//function : Formats
//purpose  : Supported formats of dcocuments
//=======================================================================
void OCAF_Application::Formats(TColStd_SequenceOfExtendedString& theFormats)
{
  theFormats.Append(TCollection_ExtendedString("MDTV-Standard"));
}
