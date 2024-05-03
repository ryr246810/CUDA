
#ifndef _OCAF_PipeShellDriver_HeaderFile
#define _OCAF_PipeShellDriver_HeaderFile


#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_PipeShellDriver,OCAF_Driver)

class OCAF_PipeShellDriver : public OCAF_Driver 
{

public:

 // Methods PUBLIC
 // 
  Standard_EXPORT OCAF_PipeShellDriver();
  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;


  DEFINE_STANDARD_RTTIEXT(OCAF_PipeShellDriver,OCAF_Driver)


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
