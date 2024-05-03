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

#ifndef _OCAF_RevolutionDriver_HeaderFile
#define _OCAF_RevolutionDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

class TFunction_Logbook;

DEFINE_STANDARD_HANDLE(OCAF_RevolutionDriver,OCAF_Driver)

class OCAF_RevolutionDriver : public OCAF_Driver 
{
public:
  // Methods PUBLIC
  // 
  Standard_EXPORT OCAF_RevolutionDriver();
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;


  DEFINE_STANDARD_RTTIEXT(OCAF_RevolutionDriver,OCAF_Driver)

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
