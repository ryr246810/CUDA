#include <TFunction_DriverTable.hxx>
#include <TPrsStd_DriverTable.hxx>

//#include <OCAF_AISFunctionDriver.hxx>

#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
#include <TDF_Label.hxx>
#include <TDF_Data.hxx>
#include <TDataStd_TreeNode.hxx>


#include "OCAF_ApplicationBase.ixx"

static Handle(TDocStd_Document) TheClipboard;

//=======================================================================
//function : GetClipboard
//purpose  : Returns static document - Clipboard
//=======================================================================
Handle(TDocStd_Document) OCAF_ApplicationBase::GetClipboard()
{
  if(TheClipboard.IsNull()) {
    TheClipboard = new TDocStd_Document("MDTV-Standard");
    TDF_Label aRootLabel = TheClipboard->GetData()->Root();
    //Set a TreeNode to the root label as the root TreeNode of the document
    Handle(TDataStd_TreeNode) aRootTreeNode = TDataStd_TreeNode::Set(aRootLabel);
    TNaming_Builder (TheClipboard->Main());
  }
  return TheClipboard;
}

//=======================================================================
//function : Constructor
//purpose  :
//=======================================================================
OCAF_ApplicationBase::OCAF_ApplicationBase()
{
 
}
