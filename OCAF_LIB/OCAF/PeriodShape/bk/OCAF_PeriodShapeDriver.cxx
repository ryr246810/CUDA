#include <CAGDDefine.hxx>

#include "OCAF_PeriodShapeDriver.ixx"
#include <OCAF_IPeriodShape.hxx>
#include <OCAF_IFunction.hxx>

#include <BRepNaming_PeriodShape.hxx>

#include <BRepAlgoAPI_Fuse.hxx>

#include <Tags.hxx>

#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>
#include <TDF_Label.hxx>
#include <TDF_ChildIterator.hxx>
#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TFunction_Function.hxx>

#include <TDataStd_Real.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDocStd_Modified.hxx>

#include <TNaming.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_Iterator.hxx>
#include <TNaming_NamedShape.hxx>

#include <TNaming_CopyShape.hxx>
#include <TColStd_IndexedDataMapOfTransientTransient.hxx>

#include <TopoDS_Shape.hxx>
#include <TopoDS.hxx>
#include <TopAbs.hxx>
#include <TopLoc_Location.hxx>

#include <gp_Vec.hxx>
#include <gp_Trsf.hxx>

#include <TopExp_Explorer.hxx>

#include <TDF_Tool.hxx>
#include <TCollection_AsciiString.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#define OK_PERIODSHAPE 0
#define PERIODSHAPE_NOT_DONE 1
#define NULL_PERIODSHAPE 2


//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_PeriodShapeDriver::OCAF_PeriodShapeDriver():OCAF_Driver() {}


//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_PeriodShapeDriver::Execute(TFunction_Logbook& theLogbook) const 
{
  //find the TranslateFunction Node
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return NULL_PERIODSHAPE;

  OCAF_IPeriodShape anInterface(aNode);
  Standard_Integer periodNum = anInterface.GetPeriodNum();

  // 1. get the context
  TopoDS_Shape theOrgShape = anInterface.GetContext();
  TopoDS_Shape currShape = theOrgShape;

  // 2.0 get the trsf of context shape
  TopLoc_Location orgLoc = theOrgShape.Location();
  gp_Trsf orgTrsf = orgLoc.Transformation();

  // 3. set the trsf distance
  Standard_Real dx = anInterface.GetDX();
  Standard_Real dy = anInterface.GetDY();
  Standard_Real dz = anInterface.GetDZ();


  for(Standard_Integer i=1; i<periodNum; i++){
    // 4. set the Location
    Standard_Real currIndx = (Standard_Real) i;
    gp_Vec aVec( (currIndx*dx), (currIndx*dy), (currIndx*dz) );
    gp_Trsf aTrsf;
    aTrsf.SetTranslation(aVec);
    TopLoc_Location aLocRes (aTrsf * orgTrsf);
    
    // 5. get the Result Shape
    TopoDS_Shape aNewShape = theOrgShape.Located(aLocRes);


    //*
    TopoDS_Shape aShapeCopy1;
    TopoDS_Shape aShapeCopy2;
    TColStd_IndexedDataMapOfTransientTransient aMapTShapes;
    TNaming_CopyShape::CopyTool(currShape, aMapTShapes, aShapeCopy1);
    TNaming_CopyShape::CopyTool(aNewShape, aMapTShapes, aShapeCopy2);
    
    Standard_Real Fuz = Precision::Confusion();
    
    BRepAlgoAPI_Fuse mkFuse;
    TopTools_ListOfShape L1, L2;
    L1.Append(aShapeCopy1);
    L2.Append(aShapeCopy2);
    mkFuse.SetArguments(L1);
    mkFuse.SetTools(L2);
    mkFuse.SetFuzzyValue(Fuz);
    mkFuse.Build();
    //*/ 

    /*
    Standard_Real Fuz = Precision::Confusion();
    BRepAlgoAPI_Fuse mkFuse;
    TopTools_ListOfShape L1, L2;
    L1.Append(currShape);
    L2.Append(aNewShape);
    mkFuse.SetArguments(L1);
    mkFuse.SetTools(L2);
    mkFuse.SetFuzzyValue(Fuz);
    mkFuse.Build();
    //*/

    //BRepAlgoAPI_Fuse mkFuse(currShape, aNewShape);
    currShape = mkFuse.Shape();
  }


  //BRepTools::Write(currShape, "tt.brep"); 

  // 6. create a child label of this driver's label
  TDF_Label aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_PeriodShape aNaming(aResultLabel);
  aNaming.Load(currShape, BRepNaming_PERIODSHAPE);

  // 8. record the evolution information using TDocStd_Modified
  // 8.1   to mark "aNode" father's label as modified 
  TDocStd_Modified::Add(aNode->Father()->Label());

  OCAF_IFunction::AddLabels(aNode, theLogbook);
  // 8.2   to mark "aNode" as modified
  theLogbook.SetImpacted(Label());
  TDocStd_Modified::Add(Label()); 
  // 8.3   to mark "aResultLabel" as modified
  theLogbook.SetImpacted(aResultLabel);
  // 8.4   to mark "aResultLabel" children as Impacted!
  TDF_ChildIterator anIterator(aResultLabel);
  for(; anIterator.More(); anIterator.Next()) {
    theLogbook.SetImpacted(anIterator.Value());
  }

  return OK_PERIODSHAPE;
}

