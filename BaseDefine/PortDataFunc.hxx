#ifndef _PortDataFunc_HeaderFile
#define _PortDataFunc_HeaderFile

#include <Standard_TypeDefine.hxx>

bool IsOpenMurPortType(const Standard_Integer theMaterialType);
bool IsInputMurPortType(const Standard_Integer theMaterialType);
bool IsInputPMLPortType(const Standard_Integer theMaterialType);
bool IsMurPortType(const Standard_Integer theMaterialType);

bool IsPECPortType(const Standard_Integer theMaterialType);
bool IsPMLPortType(const Standard_Integer theMaterialType);
bool IsOnePortType(const Standard_Integer theMaterialType, Standard_Integer& thePortType);

#endif
