
#ifndef _OCAF_Application_HeaderFile
#define _OCAF_Application_HeaderFile

#ifndef _OCAF_ApplicationBase_HeaderFile
#include <OCAF_ApplicationBase.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_Application,OCAF_ApplicationBase)

class OCAF_Application : public OCAF_ApplicationBase 
{

public:
 // Methods PUBLIC
 // 
  Standard_EXPORT OCAF_Application();
  Standard_EXPORT virtual  void InitDocument(const Handle(TDocStd_Document)& theDoc) const;
  Standard_EXPORT Standard_CString ResourcesName() ;
  Standard_EXPORT virtual  void Formats(TColStd_SequenceOfExtendedString& theFormats) ;

  
  DEFINE_STANDARD_RTTIEXT(OCAF_Application,OCAF_ApplicationBase)


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
