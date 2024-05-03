
#ifndef _BRepNaming_Cut_HeaderFile
#define _BRepNaming_Cut_HeaderFile

#ifndef _BRepNaming_BooleanOperationFeat_HeaderFile
#include <BRepNaming_BooleanOperationFeat.hxx>
#endif
class TDF_Label;
class BRepAlgoAPI_BooleanOperation;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_Cut  : public BRepNaming_BooleanOperationFeat 
{

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
  Standard_EXPORT BRepNaming_Cut();
  Standard_EXPORT BRepNaming_Cut(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Load(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  
  Standard_EXPORT   void Load(TopoDS_Shape& aShape,const BRepNaming_TypeOfPrimitive3D TypeOfResult) const;
  
  
  
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
