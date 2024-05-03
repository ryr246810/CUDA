#include <CAGDDefine.hxx>
#include <Geom_Cylinder_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_ICylinder.hxx>
#include <OCAF_IDisplayer.hxx>


#ifndef DEFAULT_CYLINDER_VALUE
#define DEFAULT_R         100.0
#define DEFAULT_H         100.0
#endif



Geom_Cylinder_TxtBuilder::
Geom_Cylinder_TxtBuilder() : Geom_TxtBuilderBase()
{
  InitVariable();
}


Geom_Cylinder_TxtBuilder::
~Geom_Cylinder_TxtBuilder()
{

}


void 
Geom_Cylinder_TxtBuilder::
InitVariable()
{
  if(!m_Node.IsNull()) m_Node.Nullify();
  if(!m_selected_Vertex_Node.IsNull()) m_selected_Vertex_Node.Nullify();
  if(!m_selected_Vector_Node.IsNull()) m_selected_Vector_Node.Nullify();

  m_R  = 100.0;
  m_H  = 100.0;

  for(Standard_Integer dir=0; dir<3; dir++){
    m_Org[dir] = 0.0;
    m_Dir[dir] = 0.0;
  }
  m_Dir[2] = 1.0;  // default z direction

  m_Type = 0;
}


void 
Geom_Cylinder_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  Geom_TxtBuilderBase::SetAttrib(tha);

  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  if(tha.hasString("type")){
    string theType = tha.getString("type");
    if(theType == "RH") m_Type = CYLINDER_R_H;
    else if(theType == "PVRH") m_Type = CYLINDER_PNT_VEC_R_H;
    else {
    } 
  }

  if(tha.hasParam("R")) m_R = tha.getParam("R");
  if(tha.hasParam("H")) m_H = tha.getParam("H");

  if(m_Type==CYLINDER_R_H){
    if(tha.hasPrmVec("org")){
      std::vector<double> tmpOrg = tha.getPrmVec("org");
      if(!tmpOrg.empty()){
	Standard_Size nb = tmpOrg.size();
	if(nb>2){
	  for(Standard_Integer dir=0; dir<3; dir++) m_Org[dir] = tmpOrg[dir];	  
	}else{
	  for(Standard_Integer n=0; n<nb; n++) m_Org[n] = tmpOrg[n];
	}
      }
    }
    if(tha.hasPrmVec("dir")){
      std::vector<double> tmpDir = tha.getPrmVec("dir");
      if(!tmpDir.empty()){
	Standard_Size nb = tmpDir.size();
	if(nb>2){
	  for(Standard_Integer dir=0; dir<3; dir++){
	    m_Dir[dir] = tmpDir[dir];
	  }	  
	}else{
	  for(Standard_Integer dir=0; dir<3; dir++){
	    m_Dir[dir] = 0.0;
	  }
	  for(Standard_Integer n=0; n<nb; n++){
	    m_Dir[n] = tmpDir[n];
	  }
	}
      }
    }
  }else if(m_Type==CYLINDER_PNT_VEC_R_H){
    if(tha.hasString("orgNodeName")){
      string theOrgNodeName = tha.getString("orgNodeName");
      m_selected_Vertex_Node = Geom_TxtBuilderBase::GetTreeNode(theOrgNodeName);
      if(m_selected_Vertex_Node.IsNull()){
	cout<<"error--------------Geom_Cylinder_TxtBuilder::SetAttrib-------------VertexNode is NULL"<<endl;
      }
    }
    if(tha.hasString("dirNodeName")){
      string theDirNodeName = tha.getString("dirNodeName");
      m_selected_Vector_Node = Geom_TxtBuilderBase::GetTreeNode(theDirNodeName);
      if(m_selected_Vector_Node.IsNull()){
	cout<<"error--------------Geom_Cylinder_TxtBuilder::SetAttrib-------------VectorNode is NULL"<<endl;
      }
    }
  }else{
    cout<<"error--------------Geom_Cylinder_TxtBuilder::SetAttrib-------------type is not set correctly"<<endl;
  }
}


void 
Geom_Cylinder_TxtBuilder::
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
    Handle_TDataStd_TreeNode aFunctionNode;
    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    aFunctionNode= OCAF_ICylinder::MakeCylinder_FunctionNode(m_Node, m_Type, anError);
    

    OCAF_ICylinder aICylinder(aFunctionNode);
    aICylinder.SetType(m_Type);

    if (m_Type == CYLINDER_R_H){
      aICylinder.SetR(m_R);
      aICylinder.SetH(m_H);
    }
    else if (m_Type ==  CYLINDER_PNT_VEC_R_H){
      aICylinder.SetR(m_R);
      aICylinder.SetH(m_H);
      aICylinder.SetPoint(m_selected_Vertex_Node);
      aICylinder.SetVector(m_selected_Vector_Node);
    }
    else{
    }

    aICylinder.MakeCylinder_Execute(anError);
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
