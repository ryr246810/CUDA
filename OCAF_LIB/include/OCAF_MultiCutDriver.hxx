
#ifndef _OCAF_MultiCutDriver_HeaderFile
#define _OCAF_MultiCutDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_MultiCutDriver,OCAF_Driver)

class OCAF_MultiCutDriver : public OCAF_Driver 
{

public:
 // Methods PUBLIC
 // 
  Standard_EXPORT OCAF_MultiCutDriver();
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;
  
  DEFINE_STANDARD_RTTIEXT(OCAF_MultiCutDriver,OCAF_Driver)

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
