#include <CAGDDefine.hxx>

#include <OCAF_IPipe.hxx>

#include <Tags.hxx>

#include <TopoDS.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepCheck_Shell.hxx>

#include <OCAF_ISelection.hxx>

#include <TDF_ChildIterator.hxx>
#include <TDF_Data.hxx>
#include <TDF_Label.hxx>
#include <TDF_Reference.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Real.hxx>
#include <TDataStd_IntegerArray.hxx>

#include <TDocStd_Modified.hxx>

#include <TFunction_Function.hxx>
#include <TFunction_Logbook.hxx>

#include <TNaming_NamedShape.hxx>

#include <Standard_ConstructionError.hxx>

#include <TColStd_Array1OfBoolean.hxx>

#include <OCAF_PipeDriver.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>
//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_IPipe::GetID() 
{
  static Standard_GUID anID("22D22E25-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IPipe::MakePipe_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
							     Standard_Integer theType,
							     TCollection_ExtendedString& theError)
{
  if (anObjectNode.IsNull() || anObjectNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }

  // 2. Construct an interface "anInterface" used for call OCAF_Object's AddFunction
  OCAF_Object anInterface(anObjectNode);
  
  // 3. To use "anInterface" to call AddFunction of OCAF_Object
  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());

  if(aFunctionNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Pipe function");
  aFunctionInterface.SetType(theType);  

  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IPipe::MakePipe_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) {
    theError = NOTDONE;
    return Standard_False;
  }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_ThruSectionDriver
  OCAF_PipeDriver aDriver;
  // 5.1  Initialize the Label of "aDriver"
  aDriver.Init(myTreeNode->Label());
  Handle(TFunction_Logbook) log;
  // 5.2  exacute the function driver
  int a = aDriver.Execute(log);
  //if(aDriver.Execute(log) > 0) {
  if(a > 0) {
    theError = ALGO_FAILED; 
    return Standard_True;
  }

  theError = DONE;
  return Standard_True;
}


//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_IPipe::OCAF_IPipe(const Handle(TDataStd_TreeNode)& aTreeNode)
  :OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) Standard_ConstructionError::Raise("Null TreeNode"); 
}


//=======================================================================
//function : GetSpine
//purpose  :
//=======================================================================
TopoDS_Wire OCAF_IPipe::GetSpine() 
{
  TopoDS_Shape aShape;
  TopoDS_Wire aResult;

  aShape = GetReference(PIPE_SPINE_TAG);

  if(!aShape.IsNull()) {
    if(aShape.ShapeType() == TopAbs_WIRE) {
      aResult = TopoDS::Wire(aShape);
    } else if(aShape.ShapeType() == TopAbs_EDGE) {
      aResult = BRepBuilderAPI_MakeWire(TopoDS::Edge(aShape)).Wire();
    }
  }
  return aResult;
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void  OCAF_IPipe::ClearSpine()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPE_SPINE_TAG);
  L.ForgetAllAttributes(Standard_True);
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void  OCAF_IPipe::ClearProfile()
{
  TDF_Label L = GetEntry().FindChild(ARGUMENTS_TAG).FindChild(PIPE_PROFILE_TAG);
  L.ForgetAllAttributes(Standard_True);
}

