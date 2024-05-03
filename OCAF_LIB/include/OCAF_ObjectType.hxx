
#ifndef _OCAF_ObjectType_HeaderFile
#define _OCAF_ObjectType_HeaderFile

enum OCAF_ObjectType { 
  OCAF_NotDefinedObjectType,

  OCAF_Box,
  OCAF_Cylinder,
  OCAF_Sphere,
  OCAF_Cone,
  OCAF_Torus,
  OCAF_Prism,
  OCAF_Revolution,

  OCAF_Polygon,
  OCAF_Circle,
  OCAF_Ellipse,
  OCAF_Parabola,
  OCAF_Arc,
  OCAF_Vector,

  OCAF_Vertex,
  OCAF_Edge,
  OCAF_Wire,
  OCAF_Face,
  OCAF_Shell,
  OCAF_Solid,

  OCAF_ThruSections,
  OCAF_PipeShell,
  OCAF_Pipe,
 
  OCAF_BRepImport,

  OCAF_Cut,
  OCAF_MultiCut,
  OCAF_Fuse,
  OCAF_MultiFuse,
  OCAF_Common,

  OCAF_Translate,
  OCAF_Rotate,
  OCAF_Mirror,
  OCAF_PeriodShape,
  OCAF_MultiRotate,

  OCAF_Selection,
  OCAF_Fillet,

  OCAF_CosPeriodEdge,
  OCAF_RecPeriodEdge,
  OCAF_HelixEdge,

  OCAF_Curve
};


#ifndef _Standard_PrimitiveTypes_HeaderFile
#include <Standard_PrimitiveTypes.hxx>
#endif

#endif
