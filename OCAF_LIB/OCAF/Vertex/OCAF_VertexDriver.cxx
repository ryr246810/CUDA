// File:	OCAF_IVertex.cxx
// Created:	2010.03.19
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>


#include "OCAF_VertexDriver.ixx"
#include <OCAF_IVertex.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Vertex.hxx>
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

#include <BRepAlgo.hxx>



#define OK_VERTEX 0
#define X_NOT_FOUND 1
#define Y_NOT_FOUND 2
#define Z_NOT_FOUND 3
#define EMPTY_VERTEX 4
#define VERTEX_NOT_DONE 5
#define NULL_VERTEX 6

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_VertexDriver::OCAF_VertexDriver():OCAF_Driver() {}


//=======================================================================
//function : getExtremaSolution
//purpose  : local function
//=======================================================================
static Standard_Boolean getExtremaSolution(OCAF_IVertex&  thePntI,
					   TopoDS_Shape& theRefShape,
					   gp_Pnt&       thePnt)
{
  gp_Pnt anInitPnt( thePntI.GetX(), thePntI.GetY(), thePntI.GetZ() );
  BRepBuilderAPI_MakeVertex mkVertex (anInitPnt);
  TopoDS_Vertex anInitV = TopoDS::Vertex(mkVertex.Shape());
  
  BRepExtrema_DistShapeShape anExt( anInitV, theRefShape );
  if ( !anExt.IsDone() || anExt.NbSolution() < 1 )
    return Standard_False;
  thePnt = anExt.PointOnShape2(1);
  Standard_Real aMinDist2 = anInitPnt.SquareDistance( thePnt );
  for ( Standard_Integer j = 2, jn = anExt.NbSolution(); j <= jn; j++ )
  {
    gp_Pnt aPnt = anExt.PointOnShape2(j);
    Standard_Real aDist2 = anInitPnt.SquareDistance( aPnt );
    if ( aDist2 > aMinDist2)
      continue;
    aMinDist2 = aDist2;
    thePnt = aPnt;
  }
  return Standard_True;
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_VertexDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_VERTEX", that is to say "no point is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_VERTEX;

  TopLoc_Location aLocation;
  Handle(TNaming_NamedShape) aPrevNS;

  // 2. create a child label "aPrevLabel" of the lable of "aNode"
  TDF_Label aPrevLabel = aNode->Label().FindChild(RESULTS_TAG);
  // 3. check whether "aPrevLabel" have a TNaming_NamedShape attribute "aPrevNS"
  if(!aPrevLabel.IsNull() && aPrevLabel.FindAttribute(TNaming_NamedShape::GetID(), aPrevNS)) {
    // 3.1 if aPrevNS is not Empty, use "aPrevNS" to set "aLocation"
    if(!aPrevNS->IsEmpty())aLocation = aPrevNS->Get().Location();
  }

  // 4. construct an instance of OCAF_IVertex "anInterface"
  OCAF_IVertex anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;

  gp_Pnt thePnt;

  if (aType == VERTEX_XYZ){
    Standard_Real X = anInterface.GetX();
    Standard_Real Y = anInterface.GetY();
    Standard_Real Z = anInterface.GetZ();
 
    thePnt = gp_Pnt(X,Y,Z);
  }
  else if(aType == VERTEX_XYZ_REF){
    Standard_Real X = anInterface.GetX();
    Standard_Real Y = anInterface.GetY();
    Standard_Real Z = anInterface.GetZ();

    TopoDS_Shape aRefShape = anInterface.GetRef();
    if (aRefShape.ShapeType() != TopAbs_VERTEX) {
      Standard_TypeMismatch::Raise("Vertex creation aborted : referenced shape is not a vertex");
    }
    gp_Pnt P = BRep_Tool::Pnt(TopoDS::Vertex(aRefShape));
    thePnt = gp_Pnt(P.X()+X, P.Y()+Y, P.Z()+Z );
  }
  else if(aType == VERTEX_CURVE_PAR){
    TopoDS_Shape aRefShape = anInterface.GetCurve();
    if (aRefShape.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Vertex On Curve creation aborted : curve shape is not an edge");
    }
    Standard_Real aFP, aLP, aP;
    Handle(Geom_Curve) aCurve = BRep_Tool::Curve(TopoDS::Edge(aRefShape), aFP, aLP);
    aP = aFP + (aLP - aFP) * anInterface.GetParameterT();
    thePnt = aCurve->Value(aP);
  }
  else if (aType == VERTEX_CURVE_COORD) { //?????????????????????????????????????????????????????????????????
    TopoDS_Shape aRefShape = anInterface.GetCurve();
    if (aRefShape.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Vertex On Curve creation aborted : curve shape is not an edge");
    }
    if (!getExtremaSolution( anInterface, aRefShape, thePnt ) ) {
      Standard_ConstructionError::Raise("Vertex On Curve creation aborted : cannot project point");
    }
  }
  else if (aType == VERTEX_SURFACE_PAR) {
    TopoDS_Shape aRefShape = anInterface.GetSurface();
    if (aRefShape.ShapeType() != TopAbs_FACE) {
      Standard_TypeMismatch::Raise("Vertex On Surface creation aborted : surface shape is not a face");
    }
    TopoDS_Face F = TopoDS::Face(aRefShape);
    Handle(Geom_Surface) aSurf = BRep_Tool::Surface(F);
    Standard_Real U1,U2,V1,V2;
    ShapeAnalysis::GetFaceUVBounds(F,U1,U2,V1,V2);
    Standard_Real U = U1 + (U2-U1) * anInterface.GetParameterU();
    Standard_Real V = V1 + (V2-V1) * anInterface.GetParameterV();
    thePnt = aSurf->Value(U,V);
  }
  else if (aType == VERTEX_SURFACE_COORD) {
    TopoDS_Shape aRefShape = anInterface.GetSurface();
    if (aRefShape.ShapeType() != TopAbs_FACE) {
      Standard_TypeMismatch::Raise("Vertex On Surface creation aborted : surface shape is not a face");
    }
    if (!getExtremaSolution( anInterface, aRefShape, thePnt ) ) {
      Standard_ConstructionError::Raise("Vertex On Surface creation aborted : cannot project point");
    }
  }
  else if (aType == VERTEX_LINES_INTERSECTION) {
    TopoDS_Shape aRefShape1 = anInterface.GetLine1();
    TopoDS_Shape aRefShape2 = anInterface.GetLine2();
    
    if (aRefShape1.ShapeType() != TopAbs_EDGE || aRefShape2.ShapeType() != TopAbs_EDGE ) {
      Standard_TypeMismatch::Raise("Creation Vertex On Lines Intersection Aborted : Line shape is not an edge");
    }
    //Calculate Lines Intersection Vertex
    BRepExtrema_DistShapeShape dst (aRefShape1, aRefShape2);
    if (dst.IsDone()) {
      gp_Pnt P1, P2;
      for (int i = 1; i <= dst.NbSolution(); i++) {
	P1 = dst.PointOnShape1(i);
	P2 = dst.PointOnShape2(i);
	Standard_Real Dist = P1.Distance(P2);
	if ( Dist <= Precision::Confusion() )
	  thePnt = P1;
	else 
	  Standard_TypeMismatch::Raise ("Lines not have an Intersection Point");
      }
    }
  }
  else {
    return EMPTY_VERTEX;
  }


  // 5. make a point using the BRepNaming_Vertex method.
  BRepBuilderAPI_MakeVertex mkVertex( thePnt );
  mkVertex.Build();

  TopoDS_Shape aShape = mkVertex.Shape();

  if (!mkVertex.IsDone()) return VERTEX_NOT_DONE;
  if (aShape.IsNull()) return NULL_VERTEX;
  if (!BRepAlgo::IsValid(aShape)) return VERTEX_NOT_DONE;    //???????????????
  
  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Vertex aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_VERTEX);

  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);

  return OK_VERTEX;
}

