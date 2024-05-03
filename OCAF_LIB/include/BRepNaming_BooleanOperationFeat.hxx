
#ifndef _BRepNaming_BooleanOperationFeat_HeaderFile
#define _BRepNaming_BooleanOperationFeat_HeaderFile

#ifndef _BRepNaming_TopNaming_HeaderFile
#include <BRepNaming_TopNaming.hxx>
#endif
#ifndef _TopAbs_ShapeEnum_HeaderFile
#include <TopAbs_ShapeEnum.hxx>
#endif
#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#include <TopTools_ListOfShape.hxx>
#include <TopoDS_Shape.hxx>

#include <BRepNaming_TypeOfPrimitive3D.hxx>

//class TopoDS_Shape;

class TDF_Label;
class BRepAlgoAPI_BooleanOperation;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming_BooleanOperationFeat  : public BRepNaming_TopNaming 
{
  
public:
  
  void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  void* operator new(size_t size) 
  { 
    return Standard::Allocate(size); 
  }
  void  operator delete(void *anAddress) 
  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  // Methods PUBLIC
  // 
  Standard_EXPORT static void AddSimpleShapes (const TopoDS_Shape& theShape, 
					       TopTools_ListOfShape& theList);
  
  Standard_EXPORT   void Init(const TDF_Label& ResultLabel) ;
  Standard_EXPORT   TDF_Label ModifiedFaces() const;
  Standard_EXPORT   TDF_Label ModifiedEdges() const;
  Standard_EXPORT   TDF_Label DeletedFaces() const;
  Standard_EXPORT   TDF_Label DeletedEdges() const;
  Standard_EXPORT   TDF_Label DeletedVertices() const;
  Standard_EXPORT   TDF_Label NewShapes() const;
  Standard_EXPORT   TDF_Label Content() const;
  Standard_EXPORT   TDF_Label DeletedDegeneratedEdges() const;
  Standard_EXPORT   Standard_Boolean IsResultChanged(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  
  



protected:
  
  // Methods PROTECTED
  // 
  
  
  Standard_EXPORT BRepNaming_BooleanOperationFeat();
  Standard_EXPORT BRepNaming_BooleanOperationFeat(const TDF_Label& ResultLabel);
  Standard_EXPORT   TopAbs_ShapeEnum ShapeType(const TopoDS_Shape& theShape) const;
  Standard_EXPORT   TopoDS_Shape GetShape(const TopoDS_Shape& theShape) const;
  Standard_EXPORT   void LoadWire(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  Standard_EXPORT   void LoadShell(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  Standard_EXPORT   void LoadContent(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  Standard_EXPORT   void LoadResult(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  Standard_EXPORT   void LoadResultWithRemoveExtraEdges(BRepAlgoAPI_BooleanOperation& MS) const;
  Standard_EXPORT   void LoadDegenerated(BRepAlgoAPI_BooleanOperation& MakeShape) const;
  
  
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
