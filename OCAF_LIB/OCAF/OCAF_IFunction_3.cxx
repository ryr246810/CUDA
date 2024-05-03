#include <OCAF_IFunction.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>

#include <OCAF_ISelection.hxx>
#include <OCAF_ITransformParent.hxx>

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

#include <TPrsStd_AISPresentation.hxx>

#include <TDF_AttributeMap.hxx>
#include <TDF_MapIteratorOfAttributeMap.hxx>

#include <TCollection_AsciiString.hxx>

#include <Standard_ConstructionError.hxx>


#include <TDataStd_RealArray.hxx>
#include <TColStd_HArray1OfReal.hxx>


//=============================================================================
//function : SetOrientation
//purpose  :
//=============================================================================
Standard_Boolean 
OCAF_IFunction::
SetOrientation(Standard_Integer theOrientation)
{
  if (myTreeNode.IsNull()) return Standard_False; 
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label L1;
  L1 = L.FindChild( ORIENTATION_TAG ); 
  TDataStd_Integer::Set(L1,theOrientation);
  //TDocStd_Modified::Add(L1);
  return Standard_True; 
}

//=============================================================================
//function : GetOrientation
//purpose  :
//=============================================================================
Standard_Integer 
OCAF_IFunction::
GetOrientation() const
{
  if (myTreeNode.IsNull()) return ERROR_ORIENTATION;
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG,Standard_False);
  if (L.IsNull()) return ERROR_ORIENTATION;
  Handle(TDataStd_Integer) aInt;
  TDF_Label L1;

  L1 = L.FindChild(ORIENTATION_TAG, Standard_False);
  if (L1.IsNull()) return ERROR_ORIENTATION;
  if (!L1.FindAttribute(TDataStd_Integer::GetID(), aInt)) return ERROR_ORIENTATION;
  Standard_Integer theOrientation = aInt->Get();
  return theOrientation;
}


//=============================================================================
//function : SetType
//purpose  :
//=============================================================================
Standard_Boolean 
OCAF_IFunction::
SetType(Standard_Integer theType)
{
  TDF_Label _label = myTreeNode->Label();

  if(!_label.IsAttribute ( TDataStd_Integer::GetID() ) ){
    TDataStd_Integer::Set(_label, theType);
    return Standard_True; 
  }else{
    _label.ForgetAttribute( TDataStd_Integer::GetID());
    TDataStd_Integer::Set(_label, theType);
    return Standard_True; 
  }

  return Standard_False;
}


//=============================================================================
//function : GetType
//purpose  :
//=============================================================================
Standard_Integer 
OCAF_IFunction::
GetType() const
{
  Handle(TDataStd_Integer) aType;

  TDF_Label _label = myTreeNode->Label();

  if(!_label.FindAttribute(TDataStd_Integer::GetID(), aType)) return 0;

  return aType->Get();
}


//=============================================================================
//function : SetReal
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
SetReal(int thePosition, 
	double theValue)
{
  if(thePosition <= 0) return;
  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label anRealArgLabel  = theArgLabel.FindChild(thePosition);
  
  TDataStd_Real::Set(anRealArgLabel, theValue);
}


//=============================================================================
//function : GetReal
//purpose  :
//=============================================================================
double 
OCAF_IFunction::
GetReal(int thePosition)
{
  if(thePosition <= 0) return 0.0;

  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label anRealArgLabel  = theArgLabel.FindChild(thePosition);

  Handle(TDataStd_Real) aReal;
  if(!anRealArgLabel.FindAttribute(TDataStd_Real::GetID(), aReal)) return 0.0;

  return aReal->Get();
}


//=============================================================================
//function : SetInteger
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
SetInteger(int thePosition, 
	   Standard_Integer theValue)
{
  if(thePosition <= 0) return;
  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label anIntArgLabel  = theArgLabel.FindChild(thePosition);
  
  TDataStd_Integer::Set(anIntArgLabel, theValue);
}


//=============================================================================
//function : GetInteger
//purpose  :
//=============================================================================
Standard_Integer 
OCAF_IFunction::
GetInteger(int thePosition)
{
  if(thePosition <= 0) return 0;

  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label anIntArgLabel  = theArgLabel.FindChild(thePosition);

  Handle(TDataStd_Integer) aInteger;
  if(!anIntArgLabel.FindAttribute(TDataStd_Integer::GetID(), aInteger)) return 0;

  return aInteger->Get();
}


//=============================================================================
//function : SetRealArray
//purpose  :
//=============================================================================
void 
OCAF_IFunction::
SetRealArray (int thePosition,
	      const Handle(TColStd_HArray1OfReal)& theArray)
{
  if(thePosition <= 0) return;

  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label theChildArgLabel = theArgLabel.FindChild(thePosition);

  Handle(TDataStd_RealArray) anAttr = TDataStd_RealArray::Set(theChildArgLabel, theArray->Lower(), theArray->Upper());
  anAttr->ChangeArray(theArray);
}


//=============================================================================
//function : GetRealArray
//purpose  :
//=============================================================================
Handle(TColStd_HArray1OfReal) 
OCAF_IFunction::
GetRealArray(int thePosition)
{
  if(thePosition <= 0) return NULL;

  TDF_Label theArgLabel = GetEntry().FindChild(ARGUMENTS_TAG);
  TDF_Label theChildArgLabel  = theArgLabel.FindChild(thePosition);

  Handle(TDataStd_RealArray) aRealArray;
  if(!theChildArgLabel.FindAttribute(TDataStd_RealArray::GetID(), aRealArray)) return NULL;

  return aRealArray->Array();
}
