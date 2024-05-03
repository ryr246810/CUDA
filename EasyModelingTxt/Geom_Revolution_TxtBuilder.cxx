#include <CAGDDefine.hxx>
#include <Geom_Revolution_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_IRevolution.hxx>
#include <OCAF_IDisplayer.hxx>


#ifndef DEFAULT_CYLINDER_VALUE
#define DEFAULT_R         100.0
#define DEFAULT_H         100.0
#endif



Geom_Revolution_TxtBuilder::
Geom_Revolution_TxtBuilder() : Geom_TxtBuilderBase()
{
  InitVariable();
}


Geom_Revolution_TxtBuilder::
~Geom_Revolution_TxtBuilder()
{

}


void 
Geom_Revolution_TxtBuilder::
InitVariable()
{
  if(!m_Node.IsNull()) m_Node.Nullify();
  if(!m_BaseNode.IsNull()) m_BaseNode.Nullify();
  if(!m_AxisNode.IsNull()) m_AxisNode.Nullify();

  m_Angle  = 360.0;
  m_Type = 0;
}


void 
Geom_Revolution_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  Geom_TxtBuilderBase::SetAttrib(tha);

  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  if(tha.hasString("type")){
    string theType = tha.getString("type");
    if(theType == "oneWay") m_Type = REVOLUTION_BASE_AXIS_ANGLE;
    else if(theType == "twoWay") m_Type = REVOLUTION_BASE_AXIS_ANGLE_2WAYS;
    else {
    } 
  }else{
    m_Type = REVOLUTION_BASE_AXIS_ANGLE;
  }

  if(tha.hasParam("angle")) m_Angle = tha.getParam("angle");

  if(tha.hasString("base")){
    string theBaseNodeName = tha.getString("base");
    m_BaseNode = Geom_TxtBuilderBase::GetTreeNode(theBaseNodeName);
    if(m_BaseNode.IsNull()){
      cout<<"error--------------Geom_Revolution_TxtBuilder::SetAttrib-------------BaseNode is NULL"<<endl;
    }
  }
  if(tha.hasString("vector")){
    string theVectorNodeName = tha.getString("vector");
    m_AxisNode = Geom_TxtBuilderBase::GetTreeNode(theVectorNodeName);
    if(m_AxisNode.IsNull()){
      cout<<"error--------------Geom_Revolution_TxtBuilder::SetAttrib-------------VectorNode is NULL"<<endl;
    }
  }
}


void 
Geom_Revolution_TxtBuilder::
Build()
{
  string tmpname = m_Name;
  Standard_CString aName =  (Standard_CString) (tmpname.c_str());

  Handle(TDataStd_TreeNode) aRoot = m_DocCtrl->GetRoot();
  
  TCollection_ExtendedString anError;
  
  if(!m_DocCtrl->GetOCAFDoc()->HasOpenCommand()){
    m_DocCtrl->GetOCAFDoc()->OpenCommand();
  }

  try {
    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    Handle(TDataStd_TreeNode) aFunctionNode = OCAF_IRevolution::MakeRevolution_FunctionNode(m_Node, m_Type, anError);
    
    OCAF_IRevolution aIRevolution(aFunctionNode);
    aIRevolution.SetType(m_Type);
    
    aIRevolution.SetBase(m_BaseNode);
    aIRevolution.SetAxis(m_AxisNode);
    aIRevolution.SetAngle(m_Angle);
    
    aIRevolution.MakeRevolution_Execute(anError);
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
