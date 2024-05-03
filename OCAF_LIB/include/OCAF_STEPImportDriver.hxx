
#ifndef _OCAF_STEPImportDriver_HeaderFile
#define _OCAF_STEPImportDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_STEPImportDriver,OCAF_Driver)

class OCAF_STEPImportDriver : public OCAF_Driver 
{

public:
  Standard_EXPORT OCAF_STEPImportDriver();
  
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;
  
  
  DEFINE_STANDARD_RTTIEXT(OCAF_STEPImportDriver,OCAF_Driver)

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
