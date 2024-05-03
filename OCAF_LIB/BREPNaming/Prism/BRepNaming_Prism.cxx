#include "BRepNaming_Prism.ixx"
#include <BRepNaming_Loader.hxx>
#include <TNaming_Builder.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_DataMapOfShapeShape.hxx>
#include <TDF_Label.hxx>
#include <TDF_TagSource.hxx>
#include <Standard_NullObject.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopExp.hxx>
#include <TColStd_ListOfInteger.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>

#ifdef MDTV_DEB
#include <TDataStd_Name.hxx>
#endif

//=======================================================================
//function : BRepNaming_Prism
//purpose  : 
//=======================================================================

BRepNaming_Prism::BRepNaming_Prism() {}

//=======================================================================
//function : BRepNaming_Prism
//purpose  : 
//=======================================================================

BRepNaming_Prism::BRepNaming_Prism(const TDF_Label& Label):BRepNaming_TopNaming(Label) {}

//=======================================================================
//function : Init
//purpose  : 
//=======================================================================

void BRepNaming_Prism::Init(const TDF_Label& Label) {
  if(Label.IsNull())
    Standard_NullObject::Raise("BRepNaming_Prism::Init The Result label is Null ..."); 
  myResultLabel = Label;
}

//=======================================================================
//function : Bottom
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Bottom() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Bottom");
#endif
  return L;
}

//=======================================================================
//function : Top
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Top() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Top");
#endif
  return L;
}

//=======================================================================
//function : Lateral
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Lateral() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Lateral");
#endif
  return L;
}

//=======================================================================
//function : Degenerated
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Degenerated() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Degenerated");
#endif
  return L;
}

//=======================================================================
//function : Dangle
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Dangle() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Dangle");
#endif
  return L;
}

//=======================================================================
//function : Content
//purpose  : 
//=======================================================================

TDF_Label BRepNaming_Prism::Content() const {
  const TDF_Label& L = ResultLabel().NewChild();
#ifdef MDTV_DEB
  TDataStd_Name::Set(L, "Content");
#endif
  return L;
}

//=======================================================================
//function : Load (Prism)
//purpose  : 
//=======================================================================

void BRepNaming_Prism::Load (const TopoDS_Shape& aShape,
			     const BRepNaming_TypeOfPrimitive3D Type) const
{
  TNaming_Builder Builder (ResultLabel());   // add TNaming_NamedShape attribute to the ResultLabel
  if (Type == BRepNaming_PRISM){
    Builder.Generated (aShape);
  }
}

