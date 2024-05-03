
#include <CAGDDefine.hxx>

#include "OCAF_ThruSectionsDriver.ixx"

#include <OCAF_IThruSections.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>



#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>

#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepPrimAPI_MakePrism.hxx>

#include <BRepNaming_ThruSections.hxx>


#include <TDF_ChildIterator.hxx>
#include <TDF_AttributeMap.hxx>


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


#define OK_THRUSECTION 0
#define ALGO_NOT_DONE 1
#define EMPTY_THRUSECTION 2

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_ThruSectionsDriver::OCAF_ThruSectionsDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_ThruSectionsDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_THRUSECTION;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    if(!aPrevNS->IsEmpty())
      aLocation = aPrevNS->Get().Location();
  }

  OCAF_IThruSections anInterface(aNode);
  Standard_Boolean anIsSolid = anInterface.IsSolid();
  Standard_Boolean anIsRuled = anInterface.IsRuled();
  
  BRepOffsetAPI_ThruSections aBuilder(anIsSolid, anIsRuled, Precision::Confusion());
  TopTools_ListOfShape aList;


  TDF_AttributeMap aRefmap;
  anInterface.GetBuildThruSectionsElementsMap(aRefmap);
  Standard_Integer nbelements = aRefmap.Extent();


  for(int i = 1; i <= nbelements ; i++) {

    //========== new methods ==============>>>>
    //TopoDS_Shape aShapeSection = anInterface.GetSection(BUILD_THRUSECTION_SECTION_TAG + i);
    TopoDS_Shape aShapeSection = anInterface.GetBuildThruSectionsElement(i);

    TopAbs_ShapeEnum aTypeSect = aShapeSection.ShapeType();

    if(aTypeSect == TopAbs_WIRE){
      aBuilder.AddWire(TopoDS::Wire(aShapeSection));
      aList.Append(TopoDS::Wire(aShapeSection));
    }
    else if(aTypeSect == TopAbs_EDGE) {
      TopoDS_Edge anEdge = TopoDS::Edge(aShapeSection);
      TopoDS_Wire aWire = BRepBuilderAPI_MakeWire(anEdge);
      aBuilder.AddWire(aWire);
      aList.Append(aWire);
    }
    else if(aTypeSect == TopAbs_VERTEX) {
      TopoDS_Vertex aVert = TopoDS::Vertex(aShapeSection);
      aBuilder.AddVertex(aVert);
      aList.Append(aVert);
    }
    else{
      break;
    }
    //========== new methods ==============<<<<
  }
  aBuilder.Build();

  if(!aBuilder.IsDone())
    return ALGO_NOT_DONE;

  // Name result
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  
  BRepNaming_ThruSections aNaming(aResultLabel);
  aNaming.Load(aBuilder, aList, anIsRuled);
  
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);
  
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_THRUSECTION;
}
