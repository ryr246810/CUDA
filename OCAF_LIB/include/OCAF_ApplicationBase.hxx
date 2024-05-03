
#ifndef _OCAF_ApplicationBase_HeaderFile
#define _OCAF_ApplicationBase_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TDocStd_Application_HeaderFile
#include <TDocStd_Application.hxx>
#endif

#ifndef _TDocStd_Document_HeaderFile
#include <TDocStd_Document.hxx>
#endif

#ifndef _Standard_CString_HeaderFile
#include <Standard_CString.hxx>
#endif

#ifndef _TColStd_SequenceOfExtendedString_HeaderFile
#include <TColStd_SequenceOfExtendedString.hxx>
#endif


DEFINE_STANDARD_HANDLE(OCAF_ApplicationBase,TDocStd_Application)

class OCAF_ApplicationBase : public TDocStd_Application 
{

public:

  // Methods PUBLIC
  // 
  Standard_EXPORT static  Handle(TDocStd_Document) GetClipboard() ;
  Standard_EXPORT OCAF_ApplicationBase();
  

  DEFINE_STANDARD_RTTIEXT(OCAF_ApplicationBase,TDocStd_Application)

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
