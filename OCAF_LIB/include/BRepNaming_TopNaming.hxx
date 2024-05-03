#ifndef _BRepNaming_TopNaming_HeaderFile
#define _BRepNaming_TopNaming_HeaderFile

#ifndef _TDF_Label_HeaderFile
#include <TDF_Label.hxx>
#endif

class TDF_Label;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_TopNaming  {

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
  Standard_EXPORT inline  const TDF_Label& ResultLabel() const;
  
protected:
  // Methods PROTECTED
  Standard_EXPORT BRepNaming_TopNaming();
  Standard_EXPORT BRepNaming_TopNaming(const TDF_Label& Label);
  
  // Fields PROTECTED
  TDF_Label myResultLabel;
  
private: 
  
  // Methods PRIVATE
  
  // Fields PRIVATE
};


#include <BRepNaming_TopNaming.lxx>



// other inline functions and methods (like "C++: function call" methods)
//


#endif

