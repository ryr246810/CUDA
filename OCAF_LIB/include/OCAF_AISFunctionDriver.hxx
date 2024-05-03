
#ifndef _OCAF_AISFunctionDriver_HeaderFile
#define _OCAF_AISFunctionDriver_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TPrsStd_Driver_HeaderFile
#include <TPrsStd_Driver.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _AIS_InteractiveObject_HeaderFile
#include <AIS_InteractiveObject.hxx>
#endif

class Standard_GUID;
class TDF_Label;

DEFINE_STANDARD_HANDLE(OCAF_AISFunctionDriver,TPrsStd_Driver)

class OCAF_AISFunctionDriver : public TPrsStd_Driver 
{

public:
  // Methods PUBLIC
  // 
  Standard_EXPORT OCAF_AISFunctionDriver();

  Standard_EXPORT static const Standard_GUID& GetID() ;
  Standard_EXPORT virtual  Standard_Boolean Update(const TDF_Label& theLabel,Handle(AIS_InteractiveObject)& theAISObject) ;

  DEFINE_STANDARD_RTTIEXT(OCAF_AISFunctionDriver,TPrsStd_Driver)

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
