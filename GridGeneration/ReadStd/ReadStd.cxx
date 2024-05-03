
#include <ReadStd.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>

#include <OCAF_Application.hxx>
#include <TDocStd_Document.hxx>
#include <TDF_ChildIterator.hxx>

#include <OCAF_ColorMap.hxx>
#include <MaterialDefineInterface.hxx>


#include <sstream>
using namespace std;

#define READSTD_DBG


//=======================================================================
//function : ReadOCCStd
//purpose  :
//=======================================================================
Standard_Boolean BeginReadOCCStd()
{
  SetupColorMap();
}


//=======================================================================
//function : ReadOCCStd
//purpose  :
//=======================================================================
Standard_Boolean EndReadOCCStd()
{
  ClearColorMap();
}


//=======================================================================
//function : ReadOCCStd
//purpose  :
//=======================================================================
Standard_Boolean ReadOCCStd(const Standard_CString& SPath, Model_Ctrl*& theModelCtrl) 
{
  Standard_Boolean result = Standard_True;

  Handle(OCAF_Application) theOCAFApp = new OCAF_Application;


#if OCC_VERSION_HEX < 0x070000

#else
  BinDrivers::DefineFormat(theOCAFApp);
  XmlDrivers::DefineFormat(theOCAFApp);

  try{
    UnitsAPI::SetLocalSystem(UnitsAPI_MDTV);
  }catch (Standard_Failure)
  {
    cerr<<"Fatal Error in units initialisation"<<endl;
  }
#endif

  Handle_TDocStd_Document  theOCAFDoc;

  TCollection_ExtendedString TPath(SPath);
  try {
    theOCAFApp->Open(TPath, theOCAFDoc);
  }catch(...) {
    result = Standard_False;
  }
  if(!result) return result;

  TDF_Label theRootLabel = theOCAFDoc->GetData()->Root();
  if( !theRootLabel.IsNull()){
    TDF_ChildIterator anIterator(theRootLabel);

    for(; anIterator.More(); anIterator.Next()) {
      TDF_Label theLabel = anIterator.Value();
      Handle(TDataStd_TreeNode) aNode;

      if( !theLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode) ){
	result = Standard_False;
	break;
      }

      if(OCAF_ObjectTool::IsOneObject(aNode)){
	OCAF_Object anInterface(aNode);
	if(anInterface.GetObjectMask()>0 ){
	  TopoDS_Shape theShape = anInterface.GetObjectValue();
	  Standard_Integer theMaterialIndex = anInterface.GetObjResultMaterial(); 
	  Standard_Boolean IsOK = Standard_False;
	  Standard_Integer theMaterialType = OCAF_ColorMap::GetMaterialType(theMaterialIndex, IsOK);
	  Standard_Integer theMask = anInterface.GetObjectMask();
	  
	  if(theShape.ShapeType() == TopAbs_SOLID){
	    theModelCtrl->AppendShape(theShape, theMaterialType);
	    theModelCtrl->SetShapeMask(theShape, theMask);

#ifdef READSTD_DBG
	    cout<<"The material type index of this Shape is\t=\t"<<theMaterialIndex<<endl;
	    cout<<"The material type of this Shape is\t=\t"<<theMaterialType<<endl;
#endif

	  }
	}
      }      
    }


    TDF_ChildIterator anFaceIterator(theRootLabel);
    for(; anFaceIterator.More(); anFaceIterator.Next()) {
      TDF_Label theLabel = anFaceIterator.Value();
      Handle(TDataStd_TreeNode) aNode;
      if( !theLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode) ){
	result = Standard_False;
	break;
      }
      
      if(OCAF_ObjectTool::IsOneObject(aNode)){
	
	OCAF_Object anInterface(aNode);
	if(  anInterface.GetObjectMask()>0 ){
	  TopoDS_Shape theShape = anInterface.GetObjectValue();
	  
	  Standard_Integer theMaterialIndex = anInterface.GetObjResultMaterial(); 
	  Standard_Boolean IsOK = Standard_False;
	  Standard_Integer theMaterialType = OCAF_ColorMap::GetMaterialType(theMaterialIndex, IsOK);
	  Standard_Integer theMask = anInterface.GetObjectMask();
	  
	  if(theShape.ShapeType() == TopAbs_FACE){
	    const TopoDS_Face& theFace = TopoDS::Face(theShape);
	    theModelCtrl->AppendSpecialFace( theFace, theMaterialType);
	    theModelCtrl->SetSpecialFaceMask( theFace, theMask);
	    
#ifdef READSTD_DBG
	    cout<<"The material type index of this Face is\t"<<theMaterialIndex<<endl;
	    cout<<"The material type of this Face is\t"<<theMaterialType<<endl;
#endif
	    
	  }
	}
      }
    }
  }
  return result;
};


#undef READSTD_DBG
