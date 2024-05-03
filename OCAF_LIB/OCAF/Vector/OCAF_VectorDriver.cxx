// File:	OCAF_VectorDriver.cxx
// Created:	2010.03.19
// Author:	Wang Yue
//		<id_wangyue@hotmail.com>

#include <CAGDDefine.hxx>


#include "OCAF_VectorDriver.ixx"
#include <OCAF_IVector.hxx>
#include <OCAF_IFunction.hxx>

#include <Tags.hxx>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepExtrema_DistShapeShape.hxx>

#include <BRepNaming_Vector.hxx>
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

#include <ShapeAnalysis_Surface.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <GeomLProp_SLProps.hxx>
#include <BRepBndLib.hxx>
#include <ShapeAnalysis.hxx>

#include <Standard_TypeMismatch.hxx>
#include <Standard_NullObject.hxx>

#include <BRepAlgo.hxx>

#include <BRepLib.hxx>

#include <Geom2d_Curve.hxx>

#define OK_VECTOR 0
#define EMPTY_VECTOR 1
#define VECTOR_NOT_DONE 2
#define NULL_VECTOR 3

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_VectorDriver::OCAF_VectorDriver():OCAF_Driver() {}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================
Standard_Integer OCAF_VectorDriver::Execute(Handle(TFunction_Logbook)& theLogbook) const 
{
  // 1. check whether this driver's label(initialized from TFunction_Function's label) has a TDataStd_TreeNode attribute
  // 1.1    if have, get the tree node and set as "aNode"
  // 1.2    else return "EMPTY_VECTOR", that is to say "no box is created"
  Handle(TDataStd_TreeNode) aNode;
  if(!Label().FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode)) return EMPTY_VECTOR;


  // 4. construct an instance of OCAF_IBox "anInterface"
  OCAF_IVector anInterface(aNode);
  Standard_Integer aType = anInterface.GetType();
  TDF_Label aResultLabel;
  TopoDS_Shape aShape;

  if (aType == VECTOR_DX_DY_DZ) {
    gp_Pnt P1 = gp::Origin();
    gp_Pnt P2 (anInterface.GetDX(), anInterface.GetDY(), anInterface.GetDZ());
    if (P1.Distance(P2) < Precision::Confusion()) {
      TCollection_AsciiString aMsg ("Can not build vector with length, less than ");
      aMsg += TCollection_AsciiString(Precision::Confusion());
      Standard_ConstructionError::Raise(aMsg.ToCString());
    }
    aShape = BRepBuilderAPI_MakeEdge(P1, P2).Shape();
  } 
  else if (aType == VECTOR_TWO_PNT) {
    TopoDS_Shape aRefPnt1 = anInterface.GetPoint1();
    TopoDS_Shape aRefPnt2 = anInterface.GetPoint2();
    if (aRefPnt1.ShapeType() != TopAbs_VERTEX || aRefPnt2.ShapeType() != TopAbs_VERTEX) 
      Standard_ConstructionError::Raise("Wrong arguments: two points must be given");

    if (aRefPnt1.IsSame(aRefPnt2)) {
      Standard_ConstructionError::Raise("The end points must be different");
    }
    TopoDS_Vertex V1 = TopoDS::Vertex(aRefPnt1);
    TopoDS_Vertex V2 = TopoDS::Vertex(aRefPnt2);
    gp_Pnt P1 = BRep_Tool::Pnt(V1);
    gp_Pnt P2 = BRep_Tool::Pnt(V2);
    if (P1.Distance(P2) < Precision::Confusion()) {
      Standard_ConstructionError::Raise("The end points are too close");
    }
    aShape = BRepBuilderAPI_MakeEdge(V1, V2).Shape();
  } 
  else if(aType == VECTOR_TANGENT_CURVE_PAR) {
    TopoDS_Shape aRefCurve = anInterface.GetCurve();
    if (aRefCurve.ShapeType() != TopAbs_EDGE) {
      Standard_TypeMismatch::Raise("Tangent On Curve creation aborted : curve shape is not an edge");
    }
    Standard_Real aFParam =0., aLParam =0., aParam =0.;
    Handle(Geom_Curve) aCurve = BRep_Tool::Curve(TopoDS::Edge(aRefCurve), aFParam, aLParam);
    if(aCurve.IsNull()) {
      Standard_TypeMismatch::Raise("Tangent On Curve creation aborted : curve is null");
    }

    aParam = aFParam + (aLParam - aFParam) * anInterface.GetParameter();
    gp_Pnt aPoint1,aPoint2;
    gp_Vec aVec;
    aCurve->D1(aParam,aPoint1,aVec);
    if(aVec.Magnitude() < gp::Resolution())
      Standard_TypeMismatch::Raise("Tangent On Curve creation aborted : invalid value of tangent");
    aPoint2.SetXYZ(aPoint1.XYZ() + aVec.XYZ());
    BRepBuilderAPI_MakeEdge aBuilder(aPoint1,aPoint2);
    if(aBuilder.IsDone())
      aShape = aBuilder.Shape();
  }
  else if(aType == VECTOR_NORMAL_SURFACE_PNT){
    //===========>>> Surface
    TopoDS_Shape aRefSurface = anInterface.GetSurface();
    if (aRefSurface.IsNull()) {
      Standard_NullObject::Raise("Face for normal calculation is null");
    }
    if (aRefSurface.ShapeType() != TopAbs_FACE) {
      Standard_NullObject::Raise("Shape for normal calculation is not a face");
    }
    TopoDS_Face aFace = TopoDS::Face(aRefSurface);
    //===========<<< Surface
    
    //===========>>> Point
    gp_Pnt p1 (0,0,0);
    TopoDS_Shape anOptPnt = anInterface.GetPoint();
    if (anOptPnt.IsNull())
      Standard_NullObject::Raise("Invalid shape given for point argument");
    p1 = BRep_Tool::Pnt(TopoDS::Vertex(anOptPnt));
    //===========<<< Point
    
    // Point parameters on surface
    Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);
    Handle(ShapeAnalysis_Surface) aSurfAna = new ShapeAnalysis_Surface (aSurf);
    gp_Pnt2d pUV = aSurfAna->ValueOfUV(p1, Precision::Confusion());
    
    
    // Normal direction
    gp_Vec Vec1,Vec2;
    BRepAdaptor_Surface SF (aFace);
    SF.D1(pUV.X(), pUV.Y(), p1, Vec1, Vec2);
    gp_Vec V = Vec1.Crossed(Vec2);
    Standard_Real mod = V.Magnitude();
    if (mod < Precision::Confusion())
      Standard_NullObject::Raise("Normal vector of a face has null magnitude");
    
    
    // Set length of normal vector to average radius of curvature
    Standard_Real radius = 0.0;
    GeomLProp_SLProps aProperties (aSurf, pUV.X(), pUV.Y(), 2, Precision::Confusion());
    if (aProperties.IsCurvatureDefined()) {
      Standard_Real radius1 = Abs(aProperties.MinCurvature());
      Standard_Real radius2 = Abs(aProperties.MaxCurvature());
      if (Abs(radius1) > Precision::Confusion()) {
	radius = 1.0 / radius1;
	if (Abs(radius2) > Precision::Confusion()) {
	  radius = (radius + 1.0 / radius2) / 2.0;
	}
      }
      else {
	if (Abs(radius2) > Precision::Confusion()) {
	  radius = 1.0 / radius2;
	}
      }
    }
    
    // Set length of normal vector to average dimension of the face (only if average radius of curvature is not appropriate)
    if (radius < Precision::Confusion()) {
      Bnd_Box B;
      Standard_Real Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
      BRepBndLib::Add(aFace, B);
      B.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
      radius = ((Xmax - Xmin) + (Ymax - Ymin) + (Zmax - Zmin)) / 3.0;
    }
    
    if (radius < Precision::Confusion())
      radius = 1.0;
    
    V *= radius / mod;
    
    // consider the face orientation
    if (aFace.Orientation() == TopAbs_REVERSED || aFace.Orientation() == TopAbs_INTERNAL) {
      V = - V;
    }
    
    // Edge
    gp_Pnt p2 = p1.Translated(V);
    BRepBuilderAPI_MakeEdge aBuilder (p1, p2);
    if (!aBuilder.IsDone())
      Standard_NullObject::Raise("Vector construction failed");
    
    aShape = aBuilder.Shape();
  }
  else if(aType == VECTOR_NORMAL_SURFACE_PAR){
    //===========>>> Surface
    TopoDS_Shape aRefSurface = anInterface.GetSurface();
    if (aRefSurface.IsNull()) {
      Standard_NullObject::Raise("Face for normale calculation is null");
    }
    if (aRefSurface.ShapeType() != TopAbs_FACE) {
      Standard_NullObject::Raise("Shape for normale calculation is not a face");
    }
    TopoDS_Face aFace = TopoDS::Face(aRefSurface);
    //===========<<< Surface
    Handle(Geom_Surface) aSurf = BRep_Tool::Surface(aFace);

    // Normal direction
    gp_Pnt p1;
    gp_Vec Vec1,Vec2;
    BRepAdaptor_Surface SF (aFace);

    Standard_Real U1,U2,V1,V2;
    Standard_Real p_u,p_v;

    ShapeAnalysis::GetFaceUVBounds(aFace,U1,U2,V1,V2);
    p_u = U1 + (U2-U1) * anInterface.GetU();
    p_v = V1 + (V2-V1) * anInterface.GetV();


    SF.D1( p_u, p_v, p1, Vec1, Vec2);
    gp_Vec V = Vec1.Crossed(Vec2);
    Standard_Real mod = V.Magnitude();
    if (mod < Precision::Confusion())
      Standard_NullObject::Raise("Normal vector of a face has null magnitude");
    
    // Set length of normal vector to average radius of curvature
    Standard_Real radius = 0.0;
    GeomLProp_SLProps aProperties (aSurf, p_u , p_v , 2, Precision::Confusion());
    if (aProperties.IsCurvatureDefined()) {
      Standard_Real radius1 = Abs(aProperties.MinCurvature());
      Standard_Real radius2 = Abs(aProperties.MaxCurvature());
      if (Abs(radius1) > Precision::Confusion()) {
	radius = 1.0 / radius1;
	if (Abs(radius2) > Precision::Confusion()) {
	  radius = (radius + 1.0 / radius2) / 2.0;
	}
      }
      else {
	if (Abs(radius2) > Precision::Confusion()) {
	  radius = 1.0 / radius2;
	}
      }
    }

    // Set length of normal vector to average dimension of the face (only if average radius of curvature is not appropriate)
    if (radius < Precision::Confusion()) {
      Bnd_Box B;
      Standard_Real Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
      BRepBndLib::Add(aFace, B);
      B.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);
      radius = ((Xmax - Xmin) + (Ymax - Ymin) + (Zmax - Zmin)) / 3.0;
    }
    
    if (radius < Precision::Confusion())
      radius = 1.0;
    
    V *= radius / mod;
    
    // consider the face orientation
    if (aFace.Orientation() == TopAbs_REVERSED || aFace.Orientation() == TopAbs_INTERNAL) {
      V = - V;
    }

    // Edge
    gp_Pnt p2 = p1.Translated(V);
    BRepBuilderAPI_MakeEdge aBuilder (p1, p2);
    if (!aBuilder.IsDone())
      Standard_NullObject::Raise("Vector construction failed");
    
    aShape = aBuilder.Shape();
  }
  else{
  }

  if (aShape.IsNull()) return NULL_VECTOR;


  if (!BRepAlgo::IsValid(aShape)) return VECTOR_NOT_DONE;

  // 6. create a child label of this driver's label
  aResultLabel = Label().FindChild(RESULTS_TAG);
  // 7. append a TNaming_NamedShape attribute to "aResultLabel"
  BRepNaming_Vector aNaming(aResultLabel);
  aNaming.Load(aShape, BRepNaming_EDGE);
  
  // 8. record the evolution information using TDocStd_Modified
  OCAF_IFunction::AddLogBooks(aNode, theLogbook);

  /*
  if(!aLocation.IsIdentity()) 
    TNaming::Displace(aResultLabel, aLocation, Standard_True);
  */

  return OK_VECTOR;
}
