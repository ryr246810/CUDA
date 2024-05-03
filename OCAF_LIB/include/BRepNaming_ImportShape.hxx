
#ifndef _BRepNaming_ImportShape_HeaderFile
#define _BRepNaming_ImportShape_HeaderFile

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif

#ifndef _TDF_TagSource_HeaderFile
#include <TDF_TagSource.hxx>
#endif

#ifndef _Standard_Integer_HeaderFile
#include <Standard_Integer.hxx>
#endif

#ifndef _TDF_Label_HeaderFile
#include <TDF_Label.hxx>
#endif

#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif

#ifndef _TDF_TagSource_HeaderFile
#include <TDF_TagSource.hxx>
#endif

#ifndef _TDF_LabelMap_HeaderFile
#include <TDF_LabelMap.hxx>
#endif

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_ImportShape  : public BRepNaming_TopNaming {
  
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
  Standard_EXPORT BRepNaming_ImportShape();
  Standard_EXPORT BRepNaming_ImportShape(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load(const TopoDS_Shape& S) const;

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
