#include <OCAF_ISTEPImport.hxx>

#include <Tags.hxx>

#include <BRep_Builder.hxx>
#include <BRepTools.hxx>
#include <TDF_Label.hxx>
#include <TDocStd_Modified.hxx>
#include <TFunction_Function.hxx>
#include <TNaming_Builder.hxx>
#include <OSD_Path.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_IFunction.hxx>

#include <BRepNaming_ImportShape.hxx>
#include <TNaming_NamedShape.hxx>
#include <TDF_ChildIterator.hxx>

#include <OCAF_ObjectTool.hxx>
#include <OCAF_IDisplayer.hxx>


#include <TDF_ChildIDIterator.hxx>
#include <TDF_Label.hxx>
#include <TDataStd_Name.hxx>
#include <TDataStd_Comment.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <Interface_EntityIterator.hxx>
#include <Interface_Graph.hxx>
#include <Interface_InterfaceModel.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Reader.hxx>
#include <StepBasic_Product.hxx>
#include <StepBasic_ProductDefinition.hxx>
#include <StepBasic_ProductDefinitionFormation.hxx>
#include <StepGeom_GeometricRepresentationItem.hxx>
#include <StepShape_TopologicalRepresentationItem.hxx>
#include <StepRepr_DescriptiveRepresentationItem.hxx>
#include <StepRepr_ProductDefinitionShape.hxx>
#include <StepRepr_PropertyDefinitionRepresentation.hxx>
#include <StepRepr_Representation.hxx>
#include <TransferBRep.hxx>
#include <Transfer_TransientProcess.hxx>
#include <XSControl_TransferReader.hxx>
#include <XSControl_WorkSession.hxx>
#include <BRep_Builder.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Iterator.hxx>
#include <TColStd_SequenceOfAsciiString.hxx>

#include <StdFail_NotDone.hxx>
#include <Standard_Failure.hxx>
#include <Standard_ErrorHandler.hxx> 



namespace
{

  TopoDS_Shape 
  GetShape(const Handle(Standard_Transient) &theEnti,
	   const Handle(Transfer_TransientProcess) &theTP)
  {
    TopoDS_Shape  aResult;
    Handle(Transfer_Binder) aBinder = theTP->Find(theEnti);
    
    if (aBinder.IsNull()) {
      return aResult;
    }
    
    aResult = TransferBRep::ShapeResult(aBinder);
    return aResult;
  }

  TCollection_AsciiString 
  ToNamedUnit( const TCollection_AsciiString& unit )
  {
    TCollection_AsciiString result = unit;
    result.LowerCase();
    if ( result == "mil" ) result = "milliinch";
    return result;
  }
  
  
  TCollection_AsciiString 
  ToOcctUnit( const TCollection_AsciiString& unit, 
	      TCollection_AsciiString& error )
  {
    TCollection_AsciiString result = "M", u = ToNamedUnit(unit);
    u.LowerCase();
    
    if (u == "inch")
      result = "INCH";
    else if (u == "milliinch")
      result = "MIL";
    else if (u == "microinch")
      result = "UIN";
    else if (u == "foot")
      result = "FT";
    else if (u == "mile")
      result = "MI";
    else if (u == "metre")
      result = "M";
    else if (u == "kilometre")
      result = "KM";
    else if (u == "millimetre")
      result = "MM";
    else if (u == "centimetre")
      result = "CM";
    else if (u == "micrometre")
      result = "UM";
    else if (u.IsEmpty())
      result = "M";
    else
      error = "The file contains not supported units";
    // TODO (for other units)
    // else
    //  result = "??"
    return result;
  }
}



//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& 
OCAF_ISTEPImport::
GetID()
{
  static Standard_GUID anID("22D22E9A-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}



//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ISTEPImport::
OCAF_ISTEPImport(const Handle(TDataStd_TreeNode)& aTreeNode)  :OCAF_IFunction(aTreeNode){  }



//=======================================================================
//function : SaveFile
//purpose  :
//=======================================================================
Standard_Boolean 
OCAF_ISTEPImport::
SaveFile(const Standard_CString& theName)
{
  TopoDS_Shape aShape;

  if(myTreeNode.IsNull()) return Standard_False;
  Handle(TDataStd_TreeNode) anObjectNode = OCAF_ObjectTool::GetObjectNode(myTreeNode);
  if(anObjectNode.IsNull()) return Standard_False;	

  OCAF_Object anInterface(anObjectNode);

  aShape = anInterface.GetObjectValue();

  if (aShape.IsNull()) return Standard_False;

  return BRepTools::Write(aShape, theName);
}



//=======================================================================
//function : ReadFile
//purpose  :
//=======================================================================
void 
OCAF_ISTEPImport::
AppendFile(const TCollection_AsciiString& aNameFile, 
	   const TopoDS_Shape& theShape, 
	   const TDF_Label& theAccessLabel)
{
  Handle(TDataStd_TreeNode) anObjectNode = OCAF_ObjectTool::AddObject(theAccessLabel);
  if(anObjectNode.IsNull()) return;

  OCAF_Object anInterface(anObjectNode);
  anInterface.SetName(aNameFile);

  Handle(TDataStd_TreeNode)	aFunctionNode = anInterface.AddFunction(GetID());
  if(aFunctionNode.IsNull()) return;

  OCAF_IFunction aFuncInterface = OCAF_IFunction(aFunctionNode);
  aFuncInterface.SetName("ImportSTEP function" );

  TDF_Label aResultLabel = aFunctionNode->Label().FindChild(RESULTS_TAG);
  BRepNaming_ImportShape aNaming(aResultLabel);
  aNaming.Load(theShape);

  OCAF_IDisplayer::Display(anObjectNode);
}



void
OCAF_ISTEPImport::
CheckFile(const TCollection_AsciiString& aFileName, 
	  const TopoDS_Shape& aResShape, 
	  const TDF_Label& theAccessLabel, 
	  Standard_Integer& theShapeIndex)
{
  if (aResShape.ShapeType() == TopAbs_COMPOUND) {
    TopoDS_Iterator It (aResShape, Standard_True, Standard_True);
    for (; It.More(); It.Next()) {
      theShapeIndex++;
      TCollection_AsciiString currFileName = aFileName + "_" + theShapeIndex;
      TopoDS_Shape currShape = It.Value();

      Standard_Integer currNewShapeIndex = 0;
      CheckFile(currFileName, currShape, theAccessLabel, currNewShapeIndex);
    }
  }else{
    AppendFile(aFileName, aResShape,theAccessLabel);
  }
}



void
OCAF_ISTEPImport::
ReadFile_1(const Standard_CString& theName, 
	   const Standard_Boolean anIsIgnoreUnits,
	   const TDF_Label& theAccessLabel)
{
  if(theAccessLabel.IsNull() ) return;

  //-------------------------------------------------------------------->>GetName
  Standard_CString theInputFile = theName;
  TCollection_AsciiString aFileNameWithPath(theInputFile);
  TCollection_AsciiString aFileName = OSD_Path(aFileNameWithPath).Name();


  //-------------------------------------------------------------------->>Read StepFiles
  TCollection_AsciiString anError;

  // Set "C" numeric locale to save numbers correctly
  //Kernel_Utils::Localizer loc;

  STEPControl_Reader aReader;

  //VSR: 16/09/09: Convert to METERS
  Interface_Static::SetCVal("xstep.cascade.unit","M");
  Interface_Static::SetIVal("read.step.ideas", 1);
  Interface_Static::SetIVal("read.step.nonmanifold", 1);


  Standard_Integer theShapeIndex = 0;

  try
  {
    OCC_CATCH_SIGNALS;

    IFSelect_ReturnStatus status = aReader.ReadFile(theInputFile);
    if (status == IFSelect_RetDone) {
      // Regard or not the model units
      if( anIsIgnoreUnits ) {
        // set UnitFlag to units from file
        TColStd_SequenceOfAsciiString anUnitLengthNames;
        TColStd_SequenceOfAsciiString anUnitAngleNames;
        TColStd_SequenceOfAsciiString anUnitSolidAngleNames;
        aReader.FileUnits(anUnitLengthNames, anUnitAngleNames, anUnitSolidAngleNames);
        if (anUnitLengthNames.Length() > 0) {
          TCollection_AsciiString aLenUnits = ToOcctUnit(anUnitLengthNames.First(), anError);
          Interface_Static::SetCVal("xstep.cascade.unit", aLenUnits.ToCString());
        }
      }else{
        // Need re-scale a model (set UnitFlag to 'meter')
        Interface_Static::SetCVal("xstep.cascade.unit","M");
      }

      Standard_Boolean failsonly = Standard_False;
      aReader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity);

      // Root transfers
      Standard_Integer nbr = aReader.NbRootsForTransfer();

      aReader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity);

      for (Standard_Integer n = 1; n <= nbr; n++) {
        Standard_Boolean ok = aReader.TransferRoot(n);
        // Collecting resulting entities
        Standard_Integer nbs = aReader.NbShapes();
        if (!ok || nbs == 0){
          continue; // skip empty root
        }

        for (Standard_Integer i = 1; i <= nbs; i++) {
          TopoDS_Shape aShape = aReader.Shape(i);
	  CheckFile(aFileName,  aShape, theAccessLabel, theShapeIndex);
        }
      }
    }else{
      switch (status) {
        case IFSelect_RetVoid:
          anError = "Nothing created or No data to process";
          break;
        case IFSelect_RetError:
          anError = "Error in command or input data";
          break;
        case IFSelect_RetFail:
          anError = "Execution was run, but has failed";
          break;
        case IFSelect_RetStop:
          anError = "Execution has been stopped. Quite possible, an exception was raised";
          break;
        default:
          break;
      }
      anError = "Wrong format of the imported file. Can't import file.";
    }
  }
  catch( Standard_Failure ) {
    /*
    Handle(Standard_Failure) aFail = Standard_Failure::Caught();
    anError = aFail->GetMessageString();
    //*/
  }
}





void
OCAF_ISTEPImport::
ReadFile_2(const Standard_CString& theName, 
	   const Standard_Boolean anIsIgnoreUnits,
	   const TDF_Label& theAccessLabel)
{
  if( theAccessLabel.IsNull() ) return;

  //-------------------------------------------------------------------->>GetName
  Standard_CString theInputFile = theName;
  TCollection_AsciiString aFileNameWithPath(theInputFile);
  TCollection_AsciiString aFileName = OSD_Path(aFileNameWithPath).Name();


  TopoDS_Shape aResShape;
  TCollection_AsciiString anError;

  // Set "C" numeric locale to save numbers correctly
  //Kernel_Utils::Localizer loc;

  STEPControl_Reader aReader;

  //VSR: 16/09/09: Convert to METERS
  Interface_Static::SetCVal("xstep.cascade.unit","M");
  Interface_Static::SetIVal("read.step.ideas", 1);
  Interface_Static::SetIVal("read.step.nonmanifold", 1);

  BRep_Builder B;
  TopoDS_Compound compound;
  B.MakeCompound(compound);

  try
  {
    OCC_CATCH_SIGNALS;

    IFSelect_ReturnStatus status = aReader.ReadFile(theInputFile);

    if (status == IFSelect_RetDone) {
      // Regard or not the model units
      if( anIsIgnoreUnits ) {
        // set UnitFlag to units from file
        TColStd_SequenceOfAsciiString anUnitLengthNames;
        TColStd_SequenceOfAsciiString anUnitAngleNames;
        TColStd_SequenceOfAsciiString anUnitSolidAngleNames;
        aReader.FileUnits(anUnitLengthNames, anUnitAngleNames, anUnitSolidAngleNames);
        if (anUnitLengthNames.Length() > 0) {
          TCollection_AsciiString aLenUnits = ToOcctUnit(anUnitLengthNames.First(), anError);
          Interface_Static::SetCVal("xstep.cascade.unit", aLenUnits.ToCString());
        }
      }
      else {
        // Need re-scale a model (set UnitFlag to 'meter')
        Interface_Static::SetCVal("xstep.cascade.unit","M");
      }

      Standard_Boolean failsonly = Standard_False;
      aReader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity);

      // Root transfers
      Standard_Integer nbr = aReader.NbRootsForTransfer();

      aReader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity);


      for (Standard_Integer n = 1; n <= nbr; n++) {
        Standard_Boolean ok = aReader.TransferRoot(n);
        // Collecting resulting entities
        Standard_Integer nbs = aReader.NbShapes();
        if (!ok || nbs == 0)
          continue; // skip empty root

        // For a single entity
        else if (nbr == 1 && nbs == 1) {
          aResShape = aReader.Shape(1);
          if (aResShape.ShapeType() == TopAbs_COMPOUND) {
            int nbSub1 = 0;
            TopoDS_Shape currShape;
            TopoDS_Iterator It (aResShape, Standard_True, Standard_True);
            for (; It.More(); It.Next()) {
              nbSub1++;
              currShape = It.Value();
            }
            if (nbSub1 == 1)
              aResShape = currShape;
          }
          break;
        }

        for (Standard_Integer i = 1; i <= nbs; i++) {
          TopoDS_Shape aShape = aReader.Shape(i);
          if (aShape.IsNull()){
            continue;
	  }else{
            B.Add(compound, aShape);
	  }
        }
      }


      if( aResShape.IsNull() )
        aResShape = compound;

      // Check if any BRep entity has been read, there must be at least a vertex
      if ( !TopExp_Explorer( aResShape, TopAbs_VERTEX ).More() )
        StdFail_NotDone::Raise( "No geometrical data in the imported file." );


      // BEGIN: Store names and materials of sub-shapes from file
      TopTools_IndexedMapOfShape anIndices;
      TopExp::MapShapes(aResShape, anIndices);

      Handle(Interface_InterfaceModel) Model = aReader.WS()->Model();
      Handle(XSControl_TransferReader) TR = aReader.WS()->TransferReader();

      if (!TR.IsNull()) {
        Handle(Transfer_TransientProcess) TP = TR->TransientProcess();

        Standard_Integer nb = Model->NbEntities();


        for (Standard_Integer ie = 1; ie <= nb; ie++) {
          Handle(Standard_Transient) enti = Model->Value(ie);
          // Store names.

          StoreName(enti, anIndices, TP, theAccessLabel);
        }
      }
      // END: Store names and materials
    }
    else {
      switch (status) {
        case IFSelect_RetVoid:
          anError = "Nothing created or No data to process";
          break;
        case IFSelect_RetError:
          anError = "Error in command or input data";
          break;
        case IFSelect_RetFail:
          anError = "Execution was run, but has failed";
          break;
        case IFSelect_RetStop:
          anError = "Execution has been stopped. Quite possible, an exception was raised";
          break;
        default:
          break;
      }
      anError = "Wrong format of the imported file. Can't import file.";
      aResShape.Nullify();
    }
  }
  catch( Standard_Failure ) {
    /*
    Handle(Standard_Failure) aFail = Standard_Failure::Caught();
    anError = aFail->GetMessageString();
    //*/
    aResShape.Nullify();
  }


  if( aResShape.IsNull() ) {
    StdFail_NotDone::Raise( anError.ToCString() );
    return;
  }

  return;
}










//=============================================================================
/*!
 *  StoreName()
 */
//=============================================================================
void 
OCAF_ISTEPImport::
StoreName( const Handle(Standard_Transient)        &theEnti,
	   const TopTools_IndexedMapOfShape        &theIndices,
	   const Handle(Transfer_TransientProcess) &theTP,
	   const TDF_Label                         &theShapeLabel)
{
  Handle(TCollection_HAsciiString) aName;
  
  if (theEnti->IsKind(STANDARD_TYPE(StepShape_TopologicalRepresentationItem)) ||
      theEnti->IsKind(STANDARD_TYPE(StepGeom_GeometricRepresentationItem))) {
    aName = Handle(StepRepr_RepresentationItem)::DownCast(theEnti)->Name();
  } else {
    Handle(StepBasic_ProductDefinition) PD =
      Handle(StepBasic_ProductDefinition)::DownCast(theEnti);
    
    if (PD.IsNull() == Standard_False) {
      Handle(StepBasic_Product) Prod = PD->Formation()->OfProduct();
      aName = Prod->Name();
    }
  }
  
  bool isValidName = false;
  
  if (aName.IsNull() == Standard_False) {
    isValidName = true;
    
    if (aName->UsefullLength() < 1) {
      isValidName = false;
    } else if (aName->UsefullLength() == 4 &&
	       toupper (aName->Value(1)) == 'N' &&
	       toupper (aName->Value(2)) == 'O' &&
	       toupper (aName->Value(3)) == 'N' &&
	       toupper (aName->Value(4)) == 'E') {
      // skip 'N0NE' name
      isValidName = false;
    } else {
      // special check to pass names like "Open CASCADE STEP translator 6.9.1"
      TCollection_AsciiString aSkipName ("Open CASCADE STEP translator");
      
      if (aName->Length() >= aSkipName.Length()) {
	if (aName->String().SubString
	    (1, aSkipName.Length()).IsEqual(aSkipName)) {
	  isValidName = false;
	}
      }
    }
  }
  
  if (isValidName) {
    TCollection_ExtendedString aNameExt (aName->ToCString());
    
    // find target shape
    TopoDS_Shape S = GetShape(theEnti, theTP);
    
    if (S.IsNull()) {
      return;
    }
    
    // as PRODUCT can be included in the main shape
    // several times, we look here for all iclusions.
    Standard_Integer isub, nbSubs = theIndices.Extent();

    for (isub = 1; isub <= nbSubs; isub++) {
      TopoDS_Shape aSub = theIndices.FindKey(isub);
      if (aSub.IsPartner(S)) {
	AppendFile(aNameExt, aSub, theShapeLabel);
      }
    }
  }
}

