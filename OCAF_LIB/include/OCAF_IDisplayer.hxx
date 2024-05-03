
#ifndef _OCAF_IDisplayer_HeaderFile
#define _OCAF_IDisplayer_HeaderFile

#ifndef _TDF_Attribute_HeaderFile
#include <TDF_Attribute.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif

#ifndef _Quantity_NameOfColor_HeaderFile
#include <Quantity_NameOfColor.hxx>
#endif

#ifndef _Standard_Integer_HeaderFile
#include <Standard_Integer.hxx>
#endif

#ifndef _AIS_InteractiveContext_HeaderFile
#include <AIS_InteractiveContext.hxx>
#endif

#ifndef _TPrsStd_AISPresentation_HeaderFile
#include <TPrsStd_AISPresentation.hxx>
#endif


class TDF_Label;
class Quantity_Color;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class OCAF_IDisplayer  {
  
public:
  
  inline void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  inline void* operator new(size_t size) 
  { 
    return Standard::Allocate(size); 
  }
  inline void  operator delete(void *anAddress) 
  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static  void CheckAISVector(const Handle(TPrsStd_AISPresentation)& aPresentation, const bool isVector=false);
  Standard_EXPORT static  void Display(const Handle(TDF_Attribute)& theAttrib, const Standard_Boolean isUpdateViewer = Standard_True) ;
  Standard_EXPORT static  void DisplayVector(const Handle(TDF_Attribute)& theAttrib, const Standard_Boolean isUpdateViewer = Standard_True, const bool isVector=false) ;
  Standard_EXPORT static  void Erase(const Handle(TDF_Attribute)& theAttrib, const Standard_Boolean theRemove = Standard_False) ;
  Standard_EXPORT static  void Update(const Handle(TDF_Attribute)& theAttrib) ;
  Standard_EXPORT static  void Update(const TDF_Label& theAccessLabel) ;
  Standard_EXPORT static  void Remove(const Handle(TDF_Attribute)& theAttrib) ;
  Standard_EXPORT static  void UpdateViewer(const TDF_Label& theAccessLabel) ;
  Standard_EXPORT static  void DisplayAll(const TDF_Label& theAccessLabel, const Standard_Boolean isUpdated = Standard_False) ;
  
  Standard_EXPORT static  void SetTransparency(const Handle(TDF_Attribute)& theAttrib,const Standard_Real theValue) ;
  Standard_EXPORT static  void SetColor(const Handle(TDF_Attribute)& theAttrib,const Quantity_NameOfColor theColor) ;
  Standard_EXPORT static  void SetColor(const Handle(TDF_Attribute)& theAttrib,const Quantity_Color& theColor) ;
  Standard_EXPORT static  void SetColor(const Handle(TDF_Attribute)& theAttrib,const Standard_Integer R,const Standard_Integer G,const Standard_Integer B) ;
  
  Standard_EXPORT static  Standard_Real GetTransparency(const Handle(TDF_Attribute)& theAttrib);
  Standard_EXPORT static  Quantity_Color GetColor(const Handle(TDF_Attribute)& theAttrib);
  
  Standard_EXPORT static  void SetMode(const Handle(TDF_Attribute)& theAttrib,const Standard_Integer theMode) ;
  Standard_EXPORT static  void SetWidth(const Handle(TDF_Attribute)& theAttrib,const Standard_Real theWidth) ;
  Standard_EXPORT static  void Hilight(const Handle(TDF_Attribute)& theAttrib,Handle(AIS_InteractiveContext)& iContext) ;
  
  
  
  
  
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
