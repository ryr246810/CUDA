// File:        OCAF_IPolygon.cxx
// Created:     2010.05.31
// Author:      Wang Yue
// email        <id_wangyue@hotmail.com>


#include <CAGDDefine.hxx>


#include "OCAF_PolygonDriver.ixx"
#include <OCAF_IPolygon.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakePolygon.hxx>

#include <BRepNaming_Polygon.hxx>
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

#include <TopExp_Explorer.hxx>

#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>

#define OK_POLYGON 0
#define EMPTY_POLYGON 1
#define POLYGON_NOT_DONE 2
#define NULL_POLYGON 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_PolygonDriver::OCAF_PolygonDriver():OCAF_Driver() {}



//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_PolygonDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_POLYGON", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_POLYGON;


  // 4. construct an instance of OCAF_IPolygon "anInterface"
  OCAF_IPolygon anInterface(aNode);
  TDF_Label aResultLabel;

  TopoDS_Shape aShape;


  BRepBuilderAPI_MakePolygon aMakePoly;
  
  TDF_AttributeMap aRefmap;
  anInterface.GetBuildPolygonElements(aRefmap);
  
  Standard_Integer nbelements = aRefmap.Extent();
  Standard_Integer ind;
  for(ind = 1; ind<= nbelements; ind++){
    TopoDS_Shape aElement_i = anInterface.GetBuildPolygonElement(ind);
    if (aElement_i.IsNull()) {
      Standard_NullObject::Raise("Shape for polygon construction is null");
    }
    if( aElement_i.ShapeType() == TopAbs_VERTEX ) {
      aMakePoly.Add(TopoDS::Vertex(aElement_i));
    }
    else{
      Standard_TypeMismatch::Raise("Shape for wire construction is neither an edge nor a wire");
    }
  }
  
  if ( anInterface.GetIsClose() ) aMakePoly.Close();
  
  if (!aMakePoly.IsDone())  return POLYGON_NOT_DONE;
  aShape = aMakePoly.Wire();
  
  // 5.check the result wire
  if (aShape.IsNull())  return NULL_POLYGON;
  if ( !BRepAlgo::IsValid(aShape) ) return POLYGON_NOT_DONE;

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Polygon aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_POLYGON);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  return OK_POLYGON;
}

