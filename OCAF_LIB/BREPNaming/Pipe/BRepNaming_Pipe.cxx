
#include <CAGDDefine.hxx>

#include "BRepNaming_Pipe.ixx"
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TopTools_MapIteratorOfMapOfShape.hxx>
#include <TopTools_DataMapOfShapeListOfShape.hxx>
#include <TopTools_MapOfShape.hxx>
#include <TopExp_Explorer.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Vertex.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepCheck_Shell.hxx>
#include <BRepCheck_Wire.hxx>
#include <Standard_NullObject.hxx>

#include <TDataStd_Name.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <BRepNaming.hxx>
#include <BRepNaming_Loader.hxx>

//=======================================================================
//function : BRepNaming_Pipe
//purpose  : 
//=======================================================================

BRepNaming_Pipe::BRepNaming_Pipe()
{}

//=======================================================================
//function : BRepNaming_Pipe
//purpose  : 
//=======================================================================

BRepNaming_Pipe::BRepNaming_Pipe(const TDF_Label& theLabel)
{
  if(theLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Pipe:: The Result label is Null ..."); 
  myResultLabel = theLabel;
}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Pipe::Init(const TDF_Label& ResultLabel)
{
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Pipe::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}    

//=======================================================================
//function : Load (Pipe)
//purpose  : 
//=======================================================================

void BRepNaming_Pipe::Load (TopoDS_Shape& aShape, const BRepNaming_TypeOfPrimitive3D Type) const  
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_PIPE){
    Builder.Generated (aShape);
  }
}

