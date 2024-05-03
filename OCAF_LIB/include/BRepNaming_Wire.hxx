
#ifndef _BRepNaming_Wire_HeaderFile
#define _BRepNaming_Wire_HeaderFile

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif
#ifndef _BRepNaming_TypeOfPrimitive3D_HeaderFile
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#endif

class TDF_Label;
class BRepBuilderAPI_MakeFace;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_Wire  : public BRepNaming_TopNaming {

public:
  
  inline void* operator new(size_t,void* anAddress)  { return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress)  { if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT   BRepNaming_Wire();
  Standard_EXPORT   BRepNaming_Wire(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load( TopoDS_Wire& aWire,const BRepNaming_TypeOfPrimitive3D TypeOfResult ) const;
  
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
