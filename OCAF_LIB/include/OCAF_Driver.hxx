
#ifndef _OCAF_Driver_HeaderFile
#define _OCAF_Driver_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TFunction_Driver_HeaderFile
#include <TFunction_Driver.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _Standard_Integer_HeaderFile
#include <Standard_Integer.hxx>
#endif

#ifndef _TDF_LabelMap_HeaderFile
#include <TDF_LabelMap.hxx>
#endif

#ifndef _TFunction_Logbook_HeaderFile
#include <TFunction_Logbook.hxx>
#endif


DEFINE_STANDARD_HANDLE(OCAF_Driver,TFunction_Driver)

class OCAF_Driver : public TFunction_Driver 
{

public:
 // Methods PUBLIC
 // 
  Standard_EXPORT   void Validate(TFunction_Logbook& log) const;
  Standard_EXPORT virtual  Standard_Boolean MustExecute(const Handle(TFunction_Logbook)& log) const;
  Standard_EXPORT virtual  Standard_Integer Execute( Handle(TFunction_Logbook)& log) const;
  Standard_EXPORT virtual  Standard_Boolean Arguments(TDF_LabelMap& theArgs) const;
  Standard_EXPORT virtual  Standard_Boolean Results(TDF_LabelMap& theRes) const;


  DEFINE_STANDARD_RTTIEXT(OCAF_Driver,TFunction_Driver)

protected:

 // Methods PROTECTED
 // 
Standard_EXPORT OCAF_Driver();


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
