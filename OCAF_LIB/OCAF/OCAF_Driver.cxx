#include <Tags.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_LabelMap.hxx>
#include <TDF_MapIteratorOfLabelMap.hxx>
#include <TDF_Reference.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_DriverTable.hxx>
#include <TFunction_Function.hxx>


#include "OCAF_Driver.ixx"

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================

OCAF_Driver::OCAF_Driver():TFunction_Driver() {}

//=======================================================================
//function : Validate
//purpose  :
//=======================================================================

void OCAF_Driver::Validate(TFunction_Logbook& theLogbook) const {
  theLogbook.SetValid(Label().FindChild(RESULTS_TAG));
}

//=======================================================================
//function : MustExecute
//purpose  :
//=======================================================================

Standard_Boolean OCAF_Driver::MustExecute(const Handle(TFunction_Logbook)& theLogbook) const 
{
  const TDF_LabelMap& anImpacted = theLogbook->GetImpacted();
  
  if(anImpacted.Extent() == 0) return Standard_False;
  
  TDF_LabelMap args;
  if (!Arguments(args)) {
    return Standard_False;
  }
  
  TDF_MapIteratorOfLabelMap itr(args);
  for (; itr.More(); itr.Next()) {
    if (anImpacted.Contains(itr.Key())) {
      return Standard_True;
    }
  }
  return Standard_False;
}

//=======================================================================
//function : Execute
//purpose  :
//=======================================================================

Standard_Integer OCAF_Driver::Execute(Handle(TFunction_Logbook)& /*theLogbook*/) const {
  return 0;
}

//=======================================================================
//function : Arguments
//purpose  :
//=======================================================================
Standard_Boolean OCAF_Driver::Arguments(TDF_LabelMap& theArgs) const {

  theArgs.Clear();
  Handle(TDF_Reference) aReference;

  //1. check whether Label() (a TFunctin_Function attribute's Label) has a TDF_Reference attribute
  if(Label().FindAttribute(TDF_Reference::GetID(), aReference)){
    theArgs.Add(aReference->Get());
  }

  //2. find an argument label "ArgumentsLabel" of Label() (a TFunctin_Function attribute's Label)
  TDF_Label ArgumentsLabel = Label().FindChild(ARGUMENTS_TAG,Standard_False);
  if (ArgumentsLabel.IsNull()) return Standard_False;

  TDF_Label CurrentLabel;
  TDF_Label ReferencedLabel;
  
  //3. Iterate all child label of "ArgumentsLabel"
  TDF_ChildIterator itr(ArgumentsLabel, Standard_True);
  for (; itr.More(); itr.Next()) {
    CurrentLabel = itr.Value();

    //3.1 add the Direct arguments to "theArgs"
    theArgs.Add(CurrentLabel);

    //3.2 add the Referenced label to "theArgs"
    if (CurrentLabel.FindAttribute(TDF_Reference::GetID(), aReference)) {
      ReferencedLabel = aReference->Get();
      if (!ReferencedLabel.IsNull()) {
	theArgs.Add(ReferencedLabel);
      }
    }
  }
  return !theArgs.IsEmpty();
}

//=======================================================================
//function : Results
//purpose  :
//=======================================================================

Standard_Boolean OCAF_Driver::Results(TDF_LabelMap& theRes) const {

  theRes.Clear();

  TDF_Label ResultsLabel = Label().FindChild(RESULTS_TAG,Standard_False);
  if (ResultsLabel.IsNull()) return Standard_False;
  
  TDF_ChildIterator itr(ResultsLabel, Standard_True);
  for (; itr.More(); itr.Next()) theRes.Add(itr.Value());
  
  return !theRes.IsEmpty();
}
