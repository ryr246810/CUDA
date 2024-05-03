#include <CAGDDefine.hxx>
#include <Geom_SubFaceSelection_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <TopExp_Explorer.hxx>

#include <OCAF_ISelection.hxx>
#include <OCAF_IDisplayer.hxx>

#include <GEOMUtils.hxx>

Geom_SubFaceSelection_TxtBuilder::
Geom_SubFaceSelection_TxtBuilder()
  : Geom_TxtBuilderBase()
{
  InitVariable();
}


void 
Geom_SubFaceSelection_TxtBuilder::
InitVariable()
{
  m_Name   = "Selection";
  m_SelectionShapeType = TopAbs_FACE;

  for(Standard_Integer dir=0; dir<3; dir++){
    m_RefPnt[dir] = 0.0;
  }

  if(!m_Node.IsNull()) m_Node.Nullify();
  if(!m_ContextNode.IsNull()) m_ContextNode.Nullify();
}


void 
Geom_SubFaceSelection_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  Geom_TxtBuilderBase::SetAttrib(tha);

  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  if(tha.hasString("contextNodeName")){
    string theContextNodeName = tha.getString("contextNodeName");
    m_ContextNode = Geom_TxtBuilderBase::GetTreeNode(theContextNodeName);
    if(m_ContextNode.IsNull()){
      cout<<"error--------------Geom_SubFaceSelection_TxtBuilder-------------VertexNode is NULL"<<endl;
    }
  }

  if(tha.hasPrmVec("refPnt")){
    std::vector<double> tmpRefPnt = tha.getPrmVec("refPnt");
    if(!tmpRefPnt.empty()){
      Standard_Size nb = tmpRefPnt.size();

      if(nb>=2){
	for(Standard_Integer dir=0; dir<3; dir++) m_RefPnt[dir] = tmpRefPnt[dir];	  
      }else{
	for(Standard_Integer n=0; n<nb; n++) m_RefPnt[n] = tmpRefPnt[n];
      }
    }
  }
}


void 
Geom_SubFaceSelection_TxtBuilder::
Build()
{
  string tmpname = m_Name;
  Standard_CString aName =  (Standard_CString) (tmpname.c_str());

  OCAF_Object theContextInterface(m_ContextNode);
  TopoDS_Shape theContextShape = theContextInterface.GetObjectValue();

  BRepBuilderAPI_MakeVertex theMkv( gp_Pnt(m_RefPnt[0], m_RefPnt[1], m_RefPnt[2]) );
  TopoDS_Vertex theRefVertex = theMkv.Vertex ();

  TopoDS_Shape theFoundShape;
  Standard_Boolean isFound = Standard_False;

  TopExp_Explorer anExp;
  for ( anExp.Init(theContextShape, m_SelectionShapeType); 
	anExp.More(); 
	anExp.Next() ) {

    TopoDS_Shape theCurrentShape = anExp.Current();
    gp_Pnt aPnt1, aPnt2;
    Standard_Real aMinDist = GEOMUtils::GetMinDistance(theRefVertex, theCurrentShape, aPnt1, aPnt2);
    if(fabs(aMinDist) < Precision::Confusion()){
      isFound = Standard_True;
      theFoundShape = theCurrentShape;
      break;
    }
  }

  if(!isFound){
    cout<<"----------------------------------------------------------------select----------------0"<<endl;
    return;
  }

  Handle(TDataStd_TreeNode) aRoot = m_DocCtrl->GetRoot();
  TCollection_ExtendedString anError;
  if(!m_DocCtrl->GetOCAFDoc()->HasOpenCommand()){
    m_DocCtrl->GetOCAFDoc()->OpenCommand();
  }

  try {
    if(! OCAF_ISelection::MakeSelect_Prereq(theContextShape, aRoot) ){
      cout<<"----------------------------------------------------------------select----------------1"<<endl;
      return;
    }

    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    Handle(TDataStd_TreeNode) aFunctionNode =  OCAF_ISelection::MakeSelect_FunctionNode(m_Node, anError);

    OCAF_ISelection aISelection(aFunctionNode);

    aISelection.MakeSelect_Execute(theFoundShape, theContextShape, anError);

    //OCAF_IDisplayer::Display(m_Node);
  }
  catch(Standard_Failure) {
    m_DocCtrl->GetOCAFDoc()->AbortCommand();
    return;
  }

  m_DocCtrl->GetOCAFDoc()->CommitCommand();

  Geom_TxtBuilderBase::Build();
  Geom_TxtBuilderBase::MakeAvail(m_Name, m_Node);
}

