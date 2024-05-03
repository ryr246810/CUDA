#ifndef _BRepNaming_PipeShell_HeaderFile
#define _BRepNaming_PipeShell_HeaderFile



#ifndef _TopAbs_ShapeEnum_HeaderFile
#include <TopAbs_ShapeEnum.hxx>
#endif
#include <TDF_Label.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <TopoDS_Wire.hxx>

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif

class BRepNaming_PipeShell  : public BRepNaming_TopNaming {

public:
  void* operator new(size_t,void* anAddress)   { return anAddress;  }
  void* operator new(size_t size)  { return Standard::Allocate(size); }
  void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
 // Methods PUBLIC
 // 
  Standard_EXPORT        BRepNaming_PipeShell();
  Standard_EXPORT        BRepNaming_PipeShell(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load(BRepOffsetAPI_MakePipeShell& theMake,const TopoDS_Wire& theSpine,const TopTools_ListOfShape& theListOfISections) const;
  
  Standard_EXPORT   TDF_Label        First() const;
  Standard_EXPORT   TDF_Label        Last() const;
  Standard_EXPORT   TDF_Label        Lateral() const;
  Standard_EXPORT   TDF_Label        FreeEdges() const;
  Standard_EXPORT   Standard_Boolean GetDangleShapes(const TopTools_ListOfShape& theList,const TopAbs_ShapeEnum theGeneratedFrom,TopTools_MapOfShape& theDangles) const; 



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
