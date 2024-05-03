#ifndef _BRepNaming_ThruSections_HeaderFile
#define _BRepNaming_ThruSections_HeaderFile

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#include <TDF_Label.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_ThruSections  : public BRepNaming_TopNaming {

public:

  void* operator new(size_t,void* anAddress)   {  return anAddress;  }
  void* operator new(size_t size)   {  return Standard::Allocate(size);  }
  void  operator delete(void *anAddress)   {  if (anAddress) Standard::Free((Standard_Address&)anAddress);  }
  // Methods PUBLIC
  // 
  Standard_EXPORT BRepNaming_ThruSections();
  Standard_EXPORT BRepNaming_ThruSections(const TDF_Label& ResultLabel);
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   void Load(BRepOffsetAPI_ThruSections& theMake,const TopTools_ListOfShape& theListOfISections,const Standard_Boolean theIsRuled = Standard_False) const;
  
  
  Standard_EXPORT   TDF_Label First() const;
  Standard_EXPORT   TDF_Label Last() const;
  Standard_EXPORT   TDF_Label Lateral() const;
  Standard_EXPORT   TDF_Label FreeEdges() const;
  


protected:
  Standard_EXPORT   Standard_Boolean GetDangleShapes(const TopTools_ListOfShape& theList,const TopAbs_ShapeEnum theGeneratedFrom,TopTools_MapOfShape& theDangles) const;

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
