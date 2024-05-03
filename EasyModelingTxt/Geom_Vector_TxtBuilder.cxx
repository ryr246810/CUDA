#include <CAGDDefine.hxx>
#include <Geom_Vector_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_IVector.hxx>
#include <OCAF_IDisplayer.hxx>


Geom_Vector_TxtBuilder::
Geom_Vector_TxtBuilder() : Geom_TxtBuilderBase()
{
  InitVariable();
}


Geom_Vector_TxtBuilder::
~Geom_Vector_TxtBuilder()
{

}


void 
Geom_Vector_TxtBuilder::
InitVariable()
{
  if(!m_Node.IsNull()) m_Node.Nullify();
  if(!m_selected_Vertex1_Node.IsNull()) m_selected_Vertex1_Node.Nullify();
  if(!m_selected_Vertex2_Node.IsNull()) m_selected_Vertex2_Node.Nullify();

  for(Standard_Integer dir=0; dir<3; dir++){
    m_Dim[dir] = 0.0;
  }
  m_Dim[2] = 1.0;  // default z direction

  m_Type = 0;
}


void 
Geom_Vector_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  Geom_TxtBuilderBase::SetAttrib(tha);

  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  if(tha.hasString("type")){
    string theType = tha.getString("type");
    if(theType == "Dim") m_Type = VECTOR_DX_DY_DZ;
    else if(theType == "Two_Pnt") m_Type = VECTOR_TWO_PNT;
    else {
    } 
  }

  if(m_Type==VECTOR_DX_DY_DZ){
    if(tha.hasPrmVec("dims")){
      std::vector<double> tmpDims = tha.getPrmVec("dims");
      if(!tmpDims.empty()){
	Standard_Size nb = tmpDims.size();
	if(nb>2){
	  for(Standard_Integer dir=0; dir<3; dir++) m_Dim[dir] = tmpDims[dir];	  
	}else{
	  for(Standard_Integer n=0; n<nb; n++) m_Dim[n] = tmpDims[n];
	}
      }
    }
  }else if(m_Type==VECTOR_TWO_PNT){
    if(tha.hasString("firstPntName")){
      string theFirstPntNodeName = tha.getString("firstPntName");
      m_selected_Vertex1_Node = Geom_TxtBuilderBase::GetTreeNode(theFirstPntNodeName);
      if(m_selected_Vertex1_Node.IsNull()){
	cout<<"error--------------Geom_Vector_TxtBuilder::SetAttrib-------------VertexNode1 is NULL"<<endl;
      }
    }
    if(tha.hasString("secondPntName")){
      string theSecondPntNodeName = tha.getString("secondPntName");
      m_selected_Vertex2_Node = Geom_TxtBuilderBase::GetTreeNode(theSecondPntNodeName);
      if(m_selected_Vertex2_Node.IsNull()){
	cout<<"error--------------Geom_Vector_TxtBuilder::SetAttrib-------------VertexNode2 is NULL"<<endl;
      }
    }
  }else{
    cout<<"error--------------Geom_Vector_TxtBuilder::SetAttrib-------------type is not set correctly"<<endl;
  }
}


void 
Geom_Vector_TxtBuilder::
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
    Handle(TDataStd_TreeNode) aFunctionNode;
    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    aFunctionNode= OCAF_IVector::MakeVector_FunctionNode(m_Node, m_Type, anError);
    
    OCAF_IVector aIVector(aFunctionNode);
    aIVector.SetType(m_Type);
    
    
    if (m_Type == VECTOR_DX_DY_DZ){
      aIVector.SetDX(m_Dim[0]);
      aIVector.SetDY(m_Dim[1]);
      aIVector.SetDZ(m_Dim[2]);
    }
    else if (m_Type == VECTOR_TWO_PNT){
      aIVector.SetPoint1(m_selected_Vertex1_Node);
      aIVector.SetPoint2(m_selected_Vertex2_Node);
    }else{
      
    }
    aIVector.MakeVector_Execute(anError);
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
