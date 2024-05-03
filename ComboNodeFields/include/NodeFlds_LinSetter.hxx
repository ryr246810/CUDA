// --------------------------------------------------------------------
// File:	NodeFlds_LinSetter.hxx
// Purpose:	Go recursively through the dimensions and linearly interpolate two values of the field
// --------------------------------------------------------------------

#ifndef _NodeFlds_LinSetter_HeaderFile
#define _NodeFlds_LinSetter_HeaderFile


#include <NodeFlds_Iter.hxx>
#include <Standard_TypeDefine.hxx>

class NodeFlds_LinSetter
{
public:
  static void edgeFieldLinSetter(NodeFlds_Iter& iter, const int dynElecPhysDataIndex);
};

#endif
