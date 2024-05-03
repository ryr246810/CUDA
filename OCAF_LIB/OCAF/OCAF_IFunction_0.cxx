#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>


#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>
#include <OCAF_ITranslate.hxx>

#include <OCAF_IDisplayer.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Label.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>

#include <TDataStd_ChildNodeIterator.hxx>
#include <TDataStd_Integer.hxx>
#include <TDataStd_Name.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_UAttribute.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>

#include <TNaming_NamedShape.hxx>
#include <TNaming_Tool.hxx>
#include <TNaming_RefShape.hxx>
#include <TNaming_UsedShapes.hxx>
#include <TNaming_PtrNode.hxx>

#include <TPrsStd_AISPresentation.hxx>

#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TCollection_AsciiString.hxx>

#include <Standard_ConstructionError.hxx>



//=======================================================================
//function : SetCenter
//purpose  :
//=======================================================================

void OCAF_IFunction::SetCenter(const Standard_Real X,const Standard_Real Y,const Standard_Real Z)
{
  if (myTreeNode.IsNull()) return;

  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label L1;
  L1 = L.FindChild(X_CENTER_TAG);  TDataStd_Real::Set(L1,X);
  TDocStd_Modified::Add(L1);
  L1 = L.FindChild(Y_CENTER_TAG);  TDataStd_Real::Set(L1,Y);
  TDocStd_Modified::Add(L1);
  L1 = L.FindChild(Z_CENTER_TAG);  TDataStd_Real::Set(L1,Z);
  TDocStd_Modified::Add(L1);
}

//=======================================================================
//function : HasCenter
//purpose  :
//=======================================================================

Standard_Boolean OCAF_IFunction::HasCenter()
{
  if (myTreeNode.IsNull()) return Standard_False;

  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG,Standard_False);
  if (L.IsNull()) return Standard_False;

  Handle(TDataStd_Real) aReal;
  TDF_Label L1;

  L1 = L.FindChild(X_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return Standard_False;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return Standard_False;

  L1 = L.FindChild(Y_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return Standard_False;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return Standard_False;

  L1 = L.FindChild(Z_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return Standard_False;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return Standard_False;

  return Standard_True;
}

//=======================================================================
//function : GetCenter
//purpose  :
//=======================================================================

void OCAF_IFunction::GetCenter(Standard_Real& X,Standard_Real& Y,Standard_Real& Z)
{
  if (myTreeNode.IsNull()) return;

  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG,Standard_False);
  if (L.IsNull()) return;

  Handle(TDataStd_Real) aReal;
  TDF_Label L1;

  L1 = L.FindChild(X_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return;
  X = aReal->Get();

  L1 = L.FindChild(Y_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return;
  Y = aReal->Get();

  L1 = L.FindChild(Z_CENTER_TAG, Standard_False);
  if (L1.IsNull()) return;
  if (!L1.FindAttribute(TDataStd_Real::GetID(), aReal)) return;
  Z = aReal->Get();
}

gp_Pnt OCAF_IFunction::GetCenterPnt()
{
  Standard_Real X=0,Y=0,Z=0;
  GetCenter(X,Y,Z);
  return gp_Pnt(X,Y,Z);
}
