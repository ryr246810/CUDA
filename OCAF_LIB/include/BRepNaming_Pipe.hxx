#ifndef _BRepNaming_Pipe_HeaderFile
#define _BRepNaming_Pipe_HeaderFile



#ifndef _TopAbs_ShapeEnum_HeaderFile
#include <TopAbs_ShapeEnum.hxx>
#endif

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif

#ifndef _BRepNaming_TypeOfPrimitive3D_HeaderFile
#include <BRepNaming_TypeOfPrimitive3D.hxx>
#endif

#include <TDF_Label.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <TopoDS_Wire.hxx>


class BRepNaming_Pipe  : public BRepNaming_TopNaming {

public:
  void* operator new(size_t,void* anAddress)   { return anAddress;  }
  void* operator new(size_t size)  { return Standard::Allocate(size); }
  void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
 // Methods PUBLIC
 // 
  Standard_EXPORT        BRepNaming_Pipe();
  Standard_EXPORT        BRepNaming_Pipe(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load(TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D aType) const;

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





// other Inline functions and methods (like "C++: function call" methods)
//


#endif
