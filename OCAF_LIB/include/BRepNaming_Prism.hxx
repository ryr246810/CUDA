
#ifndef _BRepNaming_Prism_HeaderFile
#define _BRepNaming_Prism_HeaderFile

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif

#ifndef _BRepNaming_TypeOfPrimitive3D_HeaderFile
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#endif


class TDF_Label;
class BRepPrimAPI_MakePrism;
class TopoDS_Shape;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_Prism  : public BRepNaming_TopNaming {
  
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
  //    inline void  operator delete(void *anAddress, size_t size) 
  //      { 
  //        if (anAddress) Standard::Free((Standard_Address&)anAddress,size); 
  //      }
  // Methods PUBLIC
  // 
  Standard_EXPORT BRepNaming_Prism();
  Standard_EXPORT BRepNaming_Prism(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load(const TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const;
  Standard_EXPORT   TDF_Label Bottom() const;
  Standard_EXPORT   TDF_Label Top() const;
  Standard_EXPORT   TDF_Label Lateral() const;
  Standard_EXPORT   TDF_Label Degenerated() const;
  Standard_EXPORT   TDF_Label Dangle() const;
  Standard_EXPORT   TDF_Label Content() const;
  
  
  
  
  
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
