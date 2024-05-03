#ifndef _MeshInclude_Headfile
#define _MeshInclude_Headfile

/////////////////////////////////////////////
#include <Standard_ErrorHandler.hxx>
#include <Standard_Failure.hxx>
#include <Standard_NumericError.hxx>
#include <Standard_Transient.hxx>
/////////////////////////////////////////////
#include <math_Vector.hxx>
#include <math_Matrix.hxx>
/////////////////////////////////////////////
#include <BRepLib.hxx>
#include <BRep_Tool.hxx>
#include <BRep_CurveOnSurface.hxx>
#include <BRepTools.hxx>
#include <BRepBndLib.hxx>

#include <BRep_Builder.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakeShape.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeBox.hxx> 
#include <BRepPrimAPI_MakeOneAxis.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeRevolution.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepFeat_MakePrism.hxx>

#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepAdaptor_Surface.hxx>

#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Cut.hxx>

#include <BRepOffsetAPI_MakePipeShell.hxx>
////////////////////////////////////////////
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepIntCurveSurface_Inter.hxx>
#include <IntCurvesFace_ShapeIntersector.hxx>

#include <IntTools_FClass2d.hxx>

///////////////////////////////////////////
#include <gp.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>
////////////////////////////////////////////
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Line.hxx>
#include <Geom_Surface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_SurfaceOfRevolution.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>

#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAPI_IntCS.hxx>
////////////////////////////////////////////
#include <Geom2d_Ellipse.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom2d_Circle.hxx>
#include <Geom2d_Line.hxx>

#include <Geom_BezierCurve.hxx>
#include <Geom_BoundedCurve.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <Geom2dAPI_Interpolate.hxx>
#include <Geom2dAPI_PointsToBSpline.hxx>
#include <Geom2dAPI_InterCurveCurve.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeSegment.hxx>
#include <GC_MakeLine.hxx>
#include <GCE2d_MakeSegment.hxx>
#include <GCE2d_MakeCircle.hxx>
#include <gce_MakeRotation.hxx>
///////////////////GeomAdaptor////////////////
#include <GeomAdaptor_HSurface.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GeomAdaptor_HCurve.hxx>
///////////////////////////////////////////////
#include <GeomConvert.hxx>


///////////////////////////////////////////////
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopLoc_Location.hxx>
///////////////////////////////////////////////
#include <TopTools_ListOfShape.hxx>
#include <TopTools_DataMapOfShapeInteger.hxx>
#include <TopTools_DataMapOfIntegerShape.hxx>
#include <TopTools_DataMapIteratorOfDataMapOfShapeInteger.hxx>
#include <TColStd_DataMapIteratorOfDataMapOfIntegerInteger.hxx>


#include <TColStd_ListOfInteger.hxx>
#include <TColStd_ListIteratorOfListOfInteger.hxx>

#include <TColStd_DataMapOfIntegerListOfInteger.hxx>
#include <TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger.hxx>


#include <TColgp_HArray1OfPnt2d.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TColgp_Array1OfPnt.hxx>

#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_DataMapOfIntegerInteger.hxx>
///////////////////////////////////////////////


///////////////////ShapeAnalysis//////////////
#include <ShapeAnalysis.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <ShapeAnalysis_Curve.hxx>
//////////////////////////////////////////////

#include <BRepCheck_Wire.hxx>

#include <BRepMesh_IncrementalMesh.hxx>

#endif
