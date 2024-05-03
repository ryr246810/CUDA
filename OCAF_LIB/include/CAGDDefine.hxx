#ifndef CAGDDEFINE_H
#define CAGDDEFINE_H
// 3D-Viewer action

/*
enum CurrentAction3d { 

  CurAction3d_Nothing,

  CurAction3d_DynamicZooming,

  CurAction3d_WindowZooming,

  CurAction3d_DynamicPanning,

  CurAction3d_GlobalPanning,

  CurAction3d_DynamicRotation
};
//*/

enum CurrentAction3d { 
  CurAction3d_Nothing,
  CurAction3d_DynamicZooming,
  CurAction3d_DynamicPanning,
  CurAction3d_DynamicRotation
};



enum Material_Type { 
  Material_AIR,
  Material_PEC,
  Material_User_Define,
};

// CasCade ================================================================>
#include <Standard_Version.hxx>

#include <AIS_InteractiveContext.hxx>

#include <AIS_Selection.hxx>

#include <AIS_Shape.hxx>

#include <AIS_Trihedron.hxx>

#include <BRep_Builder.hxx>

#include <BRepPrimAPI_MakeBox.hxx>

#include <BRepPrimAPI_MakeCylinder.hxx>

#include <BRepPrimAPI_MakeSphere.hxx>

#include <BRepBuilderAPI_MakeVertex.hxx>


#include <BRepTools.hxx>

#include <Geom_Axis2Placement.hxx>
#include <Geom_Curve.hxx>
#include <Geom_Surface.hxx>

#include <GCE2d_MakeSegment.hxx>

#include <OSD_Path.hxx>

#include <Quantity_NameOfColor.hxx>

#include <SelectMgr_ListIteratorOfListOfFilter.hxx>

#include <SelectMgr_Filter.hxx>

#include <ShapeAnalysis.hxx>

#include <Standard.hxx>

#include <Standard_Boolean.hxx>

#include <Standard_CString.hxx>

#include <Standard_DefineHandle.hxx>

#include <Standard_ErrorHandler.hxx>

#include <Standard_GUID.hxx>

#include <Standard_IStream.hxx>

#include <Standard_Integer.hxx>

#include <Standard_Macro.hxx>

#include <Standard_NotImplemented.hxx>

#include <Standard_OStream.hxx>

#include <Standard_Real.hxx>

#include <StdSelect_BRepOwner.hxx>

#include <TColStd_SequenceOfExtendedString.hxx>

#include <TCollection_ExtendedString.hxx>

#include <TDataStd_ChildNodeIterator.hxx>

#include <TDataStd_Name.hxx>

#include <TDataStd_Real.hxx>

#include <TDataStd_TreeNode.hxx>

#include <TDF_Attribute.hxx>

#include <TDF_ChildIterator.hxx>

#include <TDF_Data.hxx>

#include <TDF_Label.hxx>

#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TDocStd_Application.hxx>

#include <TDocStd_Document.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>

#include <TFunction_DriverTable.hxx>

#include <TFunction_Logbook.hxx>

#include <TNaming_Builder.hxx>

#include <TNaming_NamedShape.hxx>

#include <TopAbs.hxx>

#include <TopoDS.hxx>

#include <TopoDS_Shape.hxx>

#include <TopExp.hxx>

#include <TPrsStd_AISPresentation.hxx>

#include <TPrsStd_AISViewer.hxx>

#include <TPrsStd_DriverTable.hxx>

#include <V3d_View.hxx>

#include <V3d_Viewer.hxx>

#include <V3d_TypeOfView.hxx>

#include <Tags.hxx>


#define ZERO_MASK 0
#define FINAL_MASK 1
#define ERROR_MASK -1

#define ERROR_MATERIAL -1

#define ERROR_ORIENTATION -1


#define BOX_DIMENSIONS  1
#define BOX_TWO_PNT   2



//vector
#define VECTOR_TWO_PNT             1
#define VECTOR_DX_DY_DZ            2
#define VECTOR_TANGENT_CURVE_PAR   3
#define VECTOR_NORMAL_SURFACE_PAR  4
#define VECTOR_NORMAL_SURFACE_PNT  5


//Vertex

#define VERTEX_XYZ                1
#define VERTEX_XYZ_REF            2
#define VERTEX_CURVE_PAR          3
#define VERTEX_SURFACE_PAR        4
#define VERTEX_LINES_INTERSECTION 5
#define VERTEX_CURVE_COORD        6
#define VERTEX_SURFACE_COORD      7


//Edge
#define EDGE_TWO_PNT             1
#define EDGE_PNT_DIR             2
#define EDGE_ON_SURFACE          3

//Face
#define FACE_BY_WIRE             1
#define FACE_BY_FACE_WIRE        2
#define FACE_BY_SURFACE_WIRE     3

//Revolution
#define REVOLUTION_BASE_AXIS_ANGLE        1
#define REVOLUTION_BASE_AXIS_ANGLE_2WAYS  2


//Translate
#define TRANSLATE_DIMENSIONS         1
#define TRANSLATE_TWO_POINTS         2
#define TRANSLATE_VECTOR             3
#define TRANSLATE_VECTOR_DISTANCE    4

//Rotate
#define ROTATE                1
#define ROTATE_THREE_POINTS   2

//Mirror
#define MIRROR_PLANE    1
#define MIRROR_AXIS     2
#define MIRROR_POINT    3


//Circle
#define CIRCLE_PNT_VEC_R 1
#define CIRCLE_CENTER_TWO_PNT 2
#define CIRCLE_THREE_PNT 3


//Ellipse
#define ELLIPSE_PNT_VEC_RR 1


//Parabola
#define PARABOLA_PNT_VV_F 1

//Arc
#define CIRC_ARC_THREE_PNT         1
#define CIRC_ARC_CENTER            2
#define ELLIPSE_ARC_CENTER_TWO_PNT 3

//Sphere
#define SPHERE_R         1
#define SPHERE_PNT_R     2

//Cylinder
#define CYLINDER_R_H         1
#define CYLINDER_PNT_VEC_R_H 2

//Cone
#define CONE_R1_R2_H         1
#define CONE_PNT_VEC_R1_R2_H 2

//Torus
#define TORUS_RR         1
#define TORUS_PNT_VEC_RR 2

//PRISM
#define	PRISM_PROFILE_VECTOR	1


//PIPE
#define	PIPE_PROFILE_SPINE  1

//COS
#define	COSPERIODEDGE_POLYGON 1
#define	COSPERIODEDGE_SMOOTH  2


//REC
#define	RECPERIODEDGE  1

//HELIX
#define	HELIXEDGE  1

//PERIODSHAPE
#define	PERIODSHAPE  1

//MULTIROTATE
#define	MULTIROTATE  1

//CURVE
#define	CURVE_INTERPOLATE 1
#define	CURVE_POLYGON 2
// End CasCade <============================================================

#endif

