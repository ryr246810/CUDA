
#include <CAGDDefine.hxx>

#include "BRepNaming_Sphere.ixx"
#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TDF_Label.hxx>
#include <TDF_TagSource.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Edge.hxx>
#include <Standard_NullObject.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopExp.hxx>
#include <TColStd_ListOfInteger.hxx>
#include <BRep_Tool.hxx>
#include <TNaming_NamedShape.hxx>

#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif


//=======================================================================
//function : BRepNaming_Sphere
//purpose  : 
//=======================================================================

BRepNaming_Sphere::BRepNaming_Sphere() {}

//=======================================================================
//function : BRepNaming_Sphere
//purpose  : 
//=======================================================================

BRepNaming_Sphere::BRepNaming_Sphere(const TDF_Label& ResultLabel):BRepNaming_TopNaming(ResultLabel) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Sphere::Init(const TDF_Label& ResultLabel) {
  if(ResultLabel.IsNull())
    Standard_NullObject::Raise("BRepNaming_Sphere::Init The Result label is Null ..."); 
  myResultLabel = ResultLabel;
}   


//=======================================================================
//function : Load
//purpose  : 
//=======================================================================
void BRepNaming_Sphere::Load(TopoDS_Shape& aSolid, const BRepNaming_TypeOfPrimitive3D Type) const 
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_SOLID){
    Builder.Generated (aSolid);
  }
}
