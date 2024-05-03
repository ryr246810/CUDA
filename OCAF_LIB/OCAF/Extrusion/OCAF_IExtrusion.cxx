#include <CAGDDefine.hxx>

#include <OCAF_IExtrusion.hxx>

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

#include <OCAF_ExtrusionDriver.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>
//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_IExtrusion::GetID() 
{
  static Standard_GUID anID("22D22E26-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}


//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IExtrusion::MakeExtrusion_FunctionNode(const Handle(TDataStd_TreeNode)&  anObjectNode,
								      Standard_Integer            theType,
								      TCollection_ExtendedString& theError)
{
  if (anObjectNode.IsNull() || anObjectNode->Label().IsNull()) 
    {
      theError = NULL_ACCESS_NODE;
      return NULL;
    }
  
  // 2. Construct an interface "anInterface" used for call OCAF_Object's AddFunction
  OCAF_Object anInterface(anObjectNode);
  
  // 3. To use "anInterface" to call AddFunction of OCAF_Object
  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());
  
  if(aFunctionNode.IsNull()) 
    {
      theError = NOTDONE;
      return NULL;
    }
  
  OCAF_IFunction aFunctionInterface(aFunctionNode);
  aFunctionInterface.SetName("Extrusion function");
  aFunctionInterface.SetType(theType);  
  
  theError = DONE;
  return aFunctionNode;
}

//=======================================================================
//function :
//purpose  : Adds a point object to the document
//=======================================================================
Standard_Boolean OCAF_IExtrusion::MakeExtrusion_Execute( TCollection_ExtendedString& theError )
{
  if( myTreeNode.IsNull()) 
    {
      theError = NOTDONE;
      return Standard_False;
    }
  // 5. setup a TFunction_Function's driver "aDriver" from OCAF_ThruSectionDriver
  OCAF_ExtrusionDriver aDriver;
  
  // 5.1  Initialize the Label of "aDriver"
  aDriver.Init(myTreeNode->Label());
  Handle(TFunction_Logbook) log;
  
  // 5.2  exacute the function driver
  int a = aDriver.Execute(log);
  
  if(a > 0)
    {
      theError = ALGO_FAILED; 
      return Standard_False;
    }
  
  theError = DONE;
  return Standard_True;
}

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_IExtrusion::OCAF_IExtrusion(const Handle(TDataStd_TreeNode)& aTreeNode)
:OCAF_IFunction(aTreeNode)
{
  if(myTreeNode.IsNull()) 
    Standard_ConstructionError::Raise("Null TreeNode"); 
}

//=======================================================================
//function : SetSpine
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IExtrusion::SetVector(Handle(TDataStd_TreeNode) theObjNode)
{
  if(!SetReference(EXTRUSION_ARG_VECTOR , theObjNode))
    return Standard_False;
  return Standard_True;
}

//=======================================================================
//function : GetSpine
//purpose  :
//=======================================================================
TopoDS_Edge OCAF_IExtrusion::GetVector() 
{
  TopoDS_Shape aShape;
  TopoDS_Edge aResult;
  
  aShape = GetReference(EXTRUSION_ARG_VECTOR);
  
  if(!aShape.IsNull()) 
    {
      if(aShape.ShapeType() == TopAbs_EDGE)
	{
	  aResult = TopoDS::Edge(aShape);
	}
    }//*/
  return aResult;
}

//=======================================================================
//function : GetSpineNode
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IExtrusion::GetVectorNode() 
{
  return GetReferenceNode(EXTRUSION_ARG_VECTOR);
}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void  OCAF_IExtrusion::ClearVector()
{
	TDF_Label L = myTreeNode->Label().FindChild(ARGUMENTS_TAG).FindChild(EXTRUSION_ARG_VECTOR);
	L.ForgetAllAttributes(Standard_True);
}

//=======================================================================
//function : SetProfile
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IExtrusion::SetProfile(Handle(TDataStd_TreeNode) theObjNode)
{
	if(!SetReference(EXTRUSION_ARG_BASIS , theObjNode))
		return Standard_False;
	return Standard_True;
}

//=======================================================================
//function : GetProfile
//purpose  :
//=======================================================================
//TopoDS_Shape OCAF_IExtrusion::GetProfile() 
//{
//	TopoDS_Shape aShape;
//	//TopoDS_Wire aResult;
//
//	aShape = GetReference(EXTRUSION_ARG_BASIS);
//	return aShape;
//}

//=======================================================================
//function : GetSpineNode
//purpose  :
//=======================================================================
//Handle(TDataStd_TreeNode) OCAF_IExtrusion::GetProfileNode() 
//{
//	return GetReferenceNode(EXTRUSION_ARG_BASIS);
//}

//=======================================================================
//function :
//purpose  :
//=======================================================================
void  OCAF_IExtrusion::ClearProfile()
{
  TDF_Label L = myTreeNode->Label().FindChild(ARGUMENTS_TAG).FindChild(EXTRUSION_ARG_BASIS);
  L.ForgetAllAttributes(Standard_True);
}
