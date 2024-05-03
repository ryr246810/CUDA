#include <CAGDDefine.hxx>
#include <Geom_Face_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_IFace.hxx>
#include <OCAF_IDisplayer.hxx>


#ifndef DEFAULT_CYLINDER_VALUE
#define DEFAULT_R         100.0
#define DEFAULT_H         100.0
#endif



Geom_Face_TxtBuilder::
Geom_Face_TxtBuilder() : Geom_TxtBuilderBase()
{
  InitVariable();
}


Geom_Face_TxtBuilder::
~Geom_Face_TxtBuilder()
{

}


void 
Geom_Face_TxtBuilder::
InitVariable()
{
  if(!m_Node.IsNull()) m_Node.Nullify();
  if(!m_selected_Wire_Node.IsNull()) m_selected_Wire_Node.Nullify();
  m_IsPlanar = Standard_False;
  m_Type = FACE_BY_WIRE;
}


void 
Geom_Face_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  Geom_TxtBuilderBase::SetAttrib(tha);

  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  if(tha.hasOption("isPlanar")){
    m_IsPlanar = tha.getOption("isPlanar");
  }

  if(tha.hasString("wireName")){
    string theWireNodeName = tha.getString("wireName");
    m_selected_Wire_Node = Geom_TxtBuilderBase::GetTreeNode(theWireNodeName);
    if(m_selected_Wire_Node.IsNull()){
      cout<<"error--------------Geom_Face_TxtBuilder::SetAttrib-------------WireNode is NULL"<<endl;
      exit(1);
    }
  }else{
    cout<<"error--------------Geom_Face_TxtBuilder::SetAttrib-------------wire is not set"<<endl;
    exit(1);
  }
}


void 
Geom_Face_TxtBuilder::
Build()
{
  string tmpname = m_Name;
  Standard_CString aName =  (Standard_CString) (tmpname.c_str());
  Handle(TDataStd_TreeNode) aRoot = m_DocCtrl->GetRoot();
  TCollection_ExtendedString anError;
  if(!m_DocCtrl->GetOCAFDoc()->HasOpenCommand()){
    m_DocCtrl->GetOCAFDoc()->OpenCommand();
  }

  try{
    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    Handle(TDataStd_TreeNode) aFunctionNode = OCAF_IFace::MakeFace_FunctionNode(m_Node, m_Type, anError);
    OCAF_IFace aIFace(aFunctionNode);
    aIFace.SetType(m_Type);
    aIFace.SetIsPlanar(m_IsPlanar);
    aIFace.SetWire(m_selected_Wire_Node);
    aIFace.MakeFace_Execute(anError);
  }
  catch(Standard_Failure) {
    m_DocCtrl->GetOCAFDoc()->AbortCommand();
    return;
  }

  m_DocCtrl->GetOCAFDoc()->CommitCommand();

  Geom_TxtBuilderBase::Build();
  Geom_TxtBuilderBase::MakeAvail(m_Name, m_Node);

  OCAF_IDisplayer::Display(m_Node);
}
