//                     Copyright (C) 2010  by
//  
//                      Wang Yue, China
//  
// This software is furnished in accordance with the terms and conditions
// of the contract and with the inclusion of the above copyright notice.
// This software or any other copy thereof may not be provided or otherwise
// be made available to any other person. No title to an ownership of the
// software is hereby transferred.
//  
// At the termination of the contract, the software and all copies of this
// software must be deleted.
//

#ifndef _OCAF_FaceDriver_HeaderFile
#define _OCAF_FaceDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif


DEFINE_STANDARD_HANDLE(OCAF_FaceDriver,OCAF_Driver)

class OCAF_FaceDriver : public OCAF_Driver 
{

public:
  // Methods PUBLIC
  // 
  Standard_EXPORT OCAF_FaceDriver();
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;
  Standard_EXPORT void MakeFace (const TopoDS_Wire& theWire, const Standard_Boolean isPlanarWanted, TopoDS_Shape& theResult) const;

  DEFINE_STANDARD_RTTIEXT(OCAF_FaceDriver,OCAF_Driver)

protected:
  // Methods PROTECTED
  // 
  
  
  // Fields PROTECTED
  //
  
  
private: 
  
  // Methods PRIVATE
  // 
  
  
  // Fields PRIVATE
  //
  
  
};





// other inline functions and methods (like "C++: function call" methods)
//


#endif