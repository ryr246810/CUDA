#include <CAGDDefine.hxx>
#include <Geom_Polygon_TxtBuilder.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_ICurve.hxx>
#include <OCAF_IDisplayer.hxx>


Geom_Polygon_TxtBuilder::
Geom_Polygon_TxtBuilder() : Geom_TxtBuilderBase()
{
  InitVariable();
}


Geom_Polygon_TxtBuilder::
~Geom_Polygon_TxtBuilder()
{
  m_Pnts.clear();
}


void 
Geom_Polygon_TxtBuilder::
InitVariable()
{
  m_Pnts.clear();
  m_Type = CURVE_POLYGON;
  m_PntNum = 0;
  m_Name = "polygon";
}


void 
Geom_Polygon_TxtBuilder::
SetAttrib(const TxHierAttribSet& tha)
{
  if(tha.hasString("name")){
    m_Name = tha.getString("name");
  }

  /*
  if(tha.hasString("type")){
    string theType = tha.getString("type");
    if(theType == "Interpolate") m_Type = CURVE_INTERPOLATE;
    else if(theType == "Polygon") m_Type = CURVE_POLYGON;
    else {
    } 
  }
  //*/
  m_Type = CURVE_POLYGON;

  if(tha.hasPrmVec("pnts")){
    m_Pnts = tha.getPrmVec("pnts");
    if(!m_Pnts.empty()){
      Standard_Integer nb = (Standard_Integer)m_Pnts.size(); 
      Standard_Integer remained = nb%3;
      if(remained!=0){
	cout<<"error--------------Geom_Polygon_TxtBuilder::SetAttrib-------------pnts is not set properly"<<endl;
	exit(1);
      }else{
	m_PntNum = nb/3;
      }
    }else{
      cout<<"error--------------Geom_Polygon_TxtBuilder::SetAttrib-------------pnts is empty"<<endl;
      exit(1);
    }
  }
}



  
void 
Geom_Polygon_TxtBuilder::
Build()
{
  cout<<"Geom_Polygon_TxtBuilder::Build()-------------------------------------------------------------------------1"<<endl;
  string tmpname = m_Name;
  Standard_CString aName =  (Standard_CString) (tmpname.c_str());
  
  Handle(TDataStd_TreeNode) aRoot = m_DocCtrl->GetRoot();
  
  TCollection_ExtendedString anError;
  
  if(!m_DocCtrl->GetOCAFDoc()->HasOpenCommand()){
    m_DocCtrl->GetOCAFDoc()->OpenCommand();
  }
  
  try {
    cout<<"Geom_Polygon_TxtBuilder::Build()-------------------------------------------------------------------------2"<<endl;
    Standard_Size nb = (Standard_Integer)m_Pnts.size(); 
    Standard_Integer dIndx_a = 1;
    Standard_Integer dIndx_z = Standard_Integer(nb); 
    
    cout<<"nb = "<<nb<<endl;

    Handle(TColStd_HArray1OfReal) myCurve = new TColStd_HArray1OfReal(dIndx_a, dIndx_z);
    
    for(int i=0; i<dIndx_z; i++){
      Standard_Integer j = dIndx_a + i;
      myCurve->SetValue (j, m_Pnts[i]);
    }
    
    m_Node = OCAF_ObjectTool::Make_ObjectNode(TCollection_ExtendedString(aName), aRoot, anError);
    Handle(TDataStd_TreeNode) aFunctionNode= OCAF_ICurve::MakeCurve_FunctionNode(m_Node, m_Type, anError);
    
    OCAF_ICurve aICurve(aFunctionNode);
    
    aICurve.SetType(m_Type);
    aICurve.SetRowCount(m_PntNum);
    
    aICurve.SetArray( myCurve );
    aICurve.MakeCurve_Execute(anError);

  }
  catch(Standard_Failure) {
    m_DocCtrl->GetOCAFDoc()->AbortCommand();
    return;
  }
  
  m_DocCtrl->GetOCAFDoc()->CommitCommand();
  
  OCAF_IDisplayer::Display(m_Node);
  Geom_TxtBuilderBase::MakeAvail(m_Name, m_Node);
}
