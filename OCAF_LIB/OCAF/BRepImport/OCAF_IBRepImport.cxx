#include <OCAF_IBRepImport.hxx>

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

//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_IBRepImport::GetID()
{
  static Standard_GUID anID("22D22E99-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}

//=======================================================================
//function : MakeBox
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IBRepImport::AddShape(const TCollection_ExtendedString& theName,
						     const TopoDS_Shape& theShape,
						     const Handle(TDataStd_TreeNode)& AccessTreeNode,
						     TCollection_ExtendedString& theError)
{
  if (AccessTreeNode.IsNull()  || AccessTreeNode->Label().IsNull()) {
    theError = NULL_ACCESS_NODE;
    return NULL;
  }

  Handle(TDataStd_TreeNode) anObjectNode = OCAF_ObjectTool::AddObject(AccessTreeNode->Label());
  if(anObjectNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  OCAF_Object anInterface(anObjectNode);
  anInterface.SetName(theName);

  Handle(TDataStd_TreeNode) aFunctionNode = anInterface.AddFunction(GetID());

  if(aFunctionNode.IsNull()) {
    theError = NOTDONE;
    return NULL;
  }

  OCAF_IFunction aFuncInterface = OCAF_IFunction(aFunctionNode);

  //anInterface.SetName(theName+ TCollection_ExtendedString("_ImportBRep function"));
  aFuncInterface.SetName("ImportBRep function");  //Modified by wangy 2010.02.27


  // Name result 
  TDF_Label aResultLabel = aFunctionNode->Label().FindChild(RESULTS_TAG);
  
  BRepNaming_ImportShape aNaming(aResultLabel);
  aNaming.Load(theShape);

  theError = DONE;
  return anObjectNode;  
}

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_IBRepImport::OCAF_IBRepImport(const Handle(TDataStd_TreeNode)& aTreeNode)  :OCAF_IFunction(aTreeNode){  }

//=======================================================================
//function : ReadFile
//purpose  :
//=======================================================================
Handle(TDataStd_TreeNode) OCAF_IBRepImport::ReadFile(const Standard_CString& theName, const TDF_Label& theAccessLabel)
{
  if(theAccessLabel.IsNull()) return NULL;

  Standard_CString theInputFileName = theName;

  TCollection_AsciiString aFile(theInputFileName);
  TCollection_AsciiString aNameFile = OSD_Path(aFile).Name();

  Handle(TDataStd_TreeNode) anObjectNode = OCAF_ObjectTool::AddObject(theAccessLabel);
  if(anObjectNode.IsNull()) return NULL;

  OCAF_Object anInterface(anObjectNode);
  anInterface.SetName(aNameFile);


  Handle(TDataStd_TreeNode)	aFunctionNode = anInterface.AddFunction(GetID());
  if(aFunctionNode.IsNull()) return NULL;


  OCAF_IFunction aFuncInterface = OCAF_IFunction(aFunctionNode);
  //anInterface.SetName(TCollection_ExtendedString(aNameFile) + TCollection_ExtendedString("_ImportBRep function") );
  aFuncInterface.SetName("ImportBRep function" );


  TopoDS_Shape aShape;	
  BRep_Builder aBuilder;
  if (!BRepTools::Read(aShape,theName,aBuilder)){
    cout<<"OCAF_IBRepImport::ReadFile:"<<endl<<"\t Fail to read Input file================>"<<endl;
    return NULL;
  }

  TDF_Label aResultLabel = aFunctionNode->Label().FindChild(RESULTS_TAG);
  BRepNaming_ImportShape aNaming(aResultLabel);
  aNaming.Load(aShape);

  return anObjectNode;
}

//=======================================================================
//function : SaveFile
//purpose  :
//=======================================================================
Standard_Boolean OCAF_IBRepImport::SaveFile(const Standard_CString& theName)
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
