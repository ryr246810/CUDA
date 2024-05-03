
#include <OCAF_IDisplayer.hxx>

#include <Tags.hxx>


#include <TDF_Label.hxx>
#include <TPrsStd_AISPresentation.hxx>
#include <TNaming_NamedShape.hxx>
#include <TPrsStd_AISViewer.hxx>
#include <TDocStd_Modified.hxx>
#include <TDF_ChildIterator.hxx>
#include <OCAF_AISFunctionDriver.hxx>


#include <OCAF_AISShape.hxx>
#include <CAGDDefine.hxx>


//=======================================================================
//function : Display
//purpose  : Displays the object located on the same label as theAttrib 
//=======================================================================
/*
void OCAF_IDisplayer::Display(const Handle(TDF_Attribute)& theAttrib,   const Standard_Boolean isUpdateViewer) 
{
  if (theAttrib.IsNull()) return;
  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;
  Handle(TPrsStd_AISPresentation) aPresentation;
  if (!ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation = TPrsStd_AISPresentation::Set( ShapeLabel, OCAF_AISFunctionDriver::GetID() );
  }
  aPresentation->Display();
  if(isUpdateViewer) TPrsStd_AISViewer::Update(ShapeLabel);
}
//*/

void OCAF_IDisplayer::Display(const Handle(TDF_Attribute)& theAttrib,   const Standard_Boolean isUpdateViewer) 
{
  if (theAttrib.IsNull()) return;
  TDF_Label ShapeLabel = theAttrib->Label();
  if (!ShapeLabel.IsNull()) {

    Handle(TPrsStd_AISPresentation) aPresentation;
    if(!ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      aPresentation = TPrsStd_AISPresentation::Set( ShapeLabel, OCAF_AISFunctionDriver::GetID() );
    }
    aPresentation->Display();
  }

  if(isUpdateViewer) TPrsStd_AISViewer::Update(ShapeLabel);
}


void OCAF_IDisplayer::CheckAISVector(const Handle(TPrsStd_AISPresentation)& aPresentation, const bool isVector)
{
  if(!aPresentation.IsNull()){
    Handle(OCAF_AISShape) InteractiveObject = Handle(OCAF_AISShape)::DownCast(aPresentation->GetAIS());
    if (!InteractiveObject.IsNull()) { 
      InteractiveObject->SetDisplayVectors(isVector);
    }
  }
}


//=======================================================================
//function : DisplayVector
//purpose  : Displays the object located on the same label as theAttrib 
//=======================================================================
void OCAF_IDisplayer::DisplayVector(const Handle(TDF_Attribute)& theAttrib, const Standard_Boolean isUpdateViewer,  const bool isVector) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label ShapeLabel = theAttrib->Label();
  if (!ShapeLabel.IsNull()) {
    Handle(TPrsStd_AISPresentation) aPresentation;
    if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      CheckAISVector(aPresentation, isVector);
    }
    //aPresentation->Display();
    aPresentation->Update();
  }

  if(isUpdateViewer) TPrsStd_AISViewer::Update(ShapeLabel);
}



//=======================================================================
//function : Erase
//purpose  : Erases in a viewer the object located on the same label as theAttrib 
//=======================================================================
void OCAF_IDisplayer::Erase(const Handle(TDF_Attribute)& theAttrib,  const Standard_Boolean theRemove) 
{
  if (theAttrib.IsNull()) return;
  
  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;
  
  Handle(TPrsStd_AISPresentation) aPresentation;
  if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->Erase(theRemove);
    TPrsStd_AISViewer::Update(ShapeLabel);
  }
}


//=======================================================================
//function : Remove
//purpose  :
//=======================================================================
void OCAF_IDisplayer::Remove(const Handle(TDF_Attribute)& theAttrib) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label aShapeLabel = theAttrib->Label();
  if (aShapeLabel.IsNull()) return;

  aShapeLabel.ForgetAttribute(TPrsStd_AISPresentation::GetID());
  TPrsStd_AISViewer::Update(aShapeLabel);
}


//=======================================================================
//function : Update
//purpose  :
//=======================================================================
void OCAF_IDisplayer::Update(const Handle(TDF_Attribute)& theAttrib) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label aShapeLabel = theAttrib->Label();
  if (aShapeLabel.IsNull()) return;

  Handle(TPrsStd_AISPresentation) aPresentation;
  if (aShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->Update();
    TPrsStd_AISViewer::Update(aShapeLabel);
  }
}


//=======================================================================
//function : Update
//purpose  :
//=======================================================================
/*
void OCAF_IDisplayer::Update(const TDF_Label& theAccessLabel) 
{
  TDF_ChildIterator anIterator(theAccessLabel.Root());
  Handle(TPrsStd_AISPresentation) aPresentation;

  for(; anIterator.More(); anIterator.Next()) {
    if (anIterator.Value().FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)){
      if(TDocStd_Modified::Contains(aPresentation->Label())){
	aPresentation->Update();
      }
    }
  }
  TPrsStd_AISViewer::Update(theAccessLabel);
}
//*/


/*
void OCAF_IDisplayer::Update(const TDF_Label& theAccessLabel) 
{
  TDF_ChildIterator anIterator(theAccessLabel.Root());
  Handle(TPrsStd_AISPresentation) aPresentation;

  for(; anIterator.More(); anIterator.Next()) {
    TDF_Label currentLabel =  anIterator.Value();

    if (currentLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      if(TDocStd_Modified::Contains(currentLabel)){
        if(aPresentation->HasOwnMode()){
	  aPresentation->Update();
	  TPrsStd_AISViewer::Update(currentLabel);
        }
      }
    }
  }
  TPrsStd_AISViewer::Update(theAccessLabel);
}
//*/


//*
void OCAF_IDisplayer::Update(const TDF_Label& theAccessLabel) 
{
  TDF_ChildIterator anIterator(theAccessLabel.Root());
  Handle(TPrsStd_AISPresentation) aPresentation;

  for(; anIterator.More(); anIterator.Next()) {
    TDF_Label currentLabel =  anIterator.Value();

    if (currentLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      aPresentation->Update();
      TPrsStd_AISViewer::Update(currentLabel);
    }
  }
  TPrsStd_AISViewer::Update(theAccessLabel);
}
//*/


//=======================================================================
//function : UpdateViewer
//purpose  :
//=======================================================================
//*
void OCAF_IDisplayer::UpdateViewer(const TDF_Label& theAccessLabel) 
{
  TPrsStd_AISViewer::Update(theAccessLabel);
}
//*/

/*
void OCAF_IDisplayer::UpdateViewer(const TDF_Label& theAccessLabel) 
{
  if (theAccessLabel.IsNull()) return;
  
  Handle(TPrsStd_AISPresentation) aPresentation;
  if (theAccessLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->Update();
    TPrsStd_AISViewer::Update(theAccessLabel);
  }
}
//*/

//=======================================================================
//function : DisplayAll
//purpose  :
//=======================================================================
void OCAF_IDisplayer::DisplayAll(const TDF_Label& theAccessLabel,  const Standard_Boolean isUpdated) 
{
  if(theAccessLabel.IsNull()) return;
  
  TDF_ChildIterator anIterator(theAccessLabel.Root());
  Handle(TPrsStd_AISPresentation) aPresentation;	

  for(; anIterator.More(); anIterator.Next()) {
    if(anIterator.Value().FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      aPresentation->Display(isUpdated);
      TPrsStd_AISViewer::Update(anIterator.Value());  // add 2015.05.12
    }
  }
  
  TPrsStd_AISViewer::Update(theAccessLabel);
}


//=======================================================================
//function : SetTransparency
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetTransparency(const Handle(TDF_Attribute)& theAttrib, const Standard_Real theValue) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;

  Handle(TPrsStd_AISPresentation) aPresentation;
  if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->SetTransparency(theValue);
    aPresentation->Update();
    TPrsStd_AISViewer::Update(ShapeLabel);
  }
}

//=======================================================================
//function : GetTransparency
//purpose  :
//=======================================================================
Standard_Real OCAF_IDisplayer::GetTransparency(const Handle(TDF_Attribute)& theAttrib) 
{
  Standard_Real aTransparency = 0.0;

  if (theAttrib.IsNull()) return aTransparency;

  TDF_Label aShapeLabel = theAttrib->Label();
  if (aShapeLabel.IsNull()) return aTransparency;

  Handle(TPrsStd_AISPresentation) aPresentation;
  if (aShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aTransparency = aPresentation->Transparency();
  }

  return aTransparency;
}

//=======================================================================
//function : SetColor
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetColor(const Handle(TDF_Attribute)& theAttrib, const Quantity_NameOfColor theColor) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;
  
  Handle(TPrsStd_AISPresentation) aPresentation;
  if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->SetColor(theColor);
    TPrsStd_AISViewer::Update(ShapeLabel);
  }
}

//=======================================================================
//function : SetColor
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetColor(const Handle(TDF_Attribute)& theAttrib, const Quantity_Color& theColor) 
{
  Quantity_NameOfColor aColor = theColor.Name();
  SetColor(theAttrib, aColor);
}

//=======================================================================
//function : SetColor
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetColor(const Handle(TDF_Attribute)& theAttrib,
			       const Standard_Integer R,
			       const Standard_Integer G,
			       const Standard_Integer B) 
{
  Quantity_Color aColor(R/255., G/255., B/255., Quantity_TOC_RGB);
  SetColor(theAttrib, aColor);
}


//=======================================================================
//function : GetColor
//purpose  :
//=======================================================================
Quantity_Color OCAF_IDisplayer::GetColor(const Handle(TDF_Attribute)& theAttrib) 
{
  Quantity_NameOfColor aColor = Quantity_NOC_BLACK;

  if (theAttrib.IsNull()) return aColor;

  TDF_Label aShapeLabel = theAttrib->Label();
  if (aShapeLabel.IsNull()) return aColor;

  Handle(TPrsStd_AISPresentation) aPresentation;
  if (aShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aColor = aPresentation->Color();
  }

  return aColor;
}


//=======================================================================
//function : SetMode
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetMode(const Handle(TDF_Attribute)& theAttrib, const Standard_Integer theMode)
{
  if (theAttrib.IsNull()) return;
  
  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;
  
  Handle(TPrsStd_AISPresentation) aPresentation;
  if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->SetMode(theMode);
    TPrsStd_AISViewer::Update(ShapeLabel);
  }
}

//=======================================================================
//function : SetWidth
//purpose  :
//=======================================================================
void OCAF_IDisplayer::SetWidth(const Handle(TDF_Attribute)& theAttrib, const Standard_Real theWidth) 
{
  if (theAttrib.IsNull()) return;

  TDF_Label ShapeLabel = theAttrib->Label();
  if (ShapeLabel.IsNull()) return;

  Handle(TPrsStd_AISPresentation) aPresentation;
  if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
    aPresentation->SetWidth(theWidth);
    TPrsStd_AISViewer::Update(ShapeLabel);
  }
}

//=======================================================================
//function : Hilight
//purpose  :
//=======================================================================
void OCAF_IDisplayer::Hilight(const Handle(TDF_Attribute)& theAttrib, Handle(AIS_InteractiveContext)& iContext) 
{
  if (theAttrib.IsNull()) return;
  
  //iContext->ClearCurrents(Standard_False); 
  iContext->ClearSelected(Standard_False); 

  TDF_Label ShapeLabel = theAttrib->Label();
  if (!ShapeLabel.IsNull()) {
    Handle(TPrsStd_AISPresentation) aPresentation;
    if (ShapeLabel.FindAttribute(TPrsStd_AISPresentation::GetID(), aPresentation)) {
      Handle(AIS_InteractiveObject) InteractiveObject;
      InteractiveObject = aPresentation->GetAIS();
      //iContext->AddOrRemoveCurrentObject(InteractiveObject, Standard_False);
      iContext->AddOrRemoveSelected(InteractiveObject, Standard_False);
    }
  }
  iContext->UpdateCurrentViewer();
}
