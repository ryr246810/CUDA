
#ifndef _BRepNaming_TypeOfPrimitive3D_HeaderFile
#define _BRepNaming_TypeOfPrimitive3D_HeaderFile

enum BRepNaming_TypeOfPrimitive3D { 
  BRepNaming_SELECTION,
  BRepNaming_VERTEX,
  BRepNaming_EDGE,
  BRepNaming_WIRE,
  BRepNaming_FACE,
  BRepNaming_SHELL,
  BRepNaming_SOLID,
  BRepNaming_COMPSOLID,
  BRepNaming_COMPOUND,
  BRepNaming_POLYGON,
  BRepNaming_PIPE,
  BRepNaming_REVOLUTION,
  BRepNaming_PRISM,
  BRepNaming_TRANSLATE,
  BRepNaming_ROTATE,
  BRepNaming_MULTIROTATE,
  BRepNaming_MIRROR,
  BRepNaming_COSPERIODEDGE,
  BRepNaming_RECPERIODEDGE,
  BRepNaming_HELIXEDGE,
  BRepNaming_PERIODSHAPE,
  BRepNaming_MULTIFUSE,
  BRepNaming_MULTICUT,
  BRepNaming_CUT,
  BRepNaming_FUSE,
  BRepNaming_COMMON
};


#ifndef _Standard_PrimitiveTypes_HeaderFile
#include <Standard_PrimitiveTypes.hxx>
#endif

#endif
