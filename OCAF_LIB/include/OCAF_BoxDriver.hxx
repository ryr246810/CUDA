
#ifndef _OCAF_BoxDriver_HeaderFile
#define _OCAF_BoxDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_BoxDriver,OCAF_Driver)

class OCAF_BoxDriver : public OCAF_Driver 
{

public:

  Standard_EXPORT OCAF_BoxDriver();
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;

  DEFINE_STANDARD_RTTIEXT(OCAF_BoxDriver,OCAF_Driver)


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
