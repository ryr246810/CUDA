#include <OCAFDocumentCtrl.hxx>

#define DEFAULT_UNDO_LIMIT 10


static std::string xmlFileTypeExtention(".xml");
//static std::string xmlFileFilter("xml Files (*.xml)");
static std::string xmlFileFormat("XmlOcaf");


#if OCC_VERSION_HEX < 0x070000
static std::string binFileTypeExtention(".std");
//static std::string binFileFilter("STD Files (*.std)");
static std::string binFileFormat("MDTV-Standard");
#else
static std::string binFileTypeExtention(".cbf");
//static std::string binFileFilter("cbf Files (*.cbf)");
static std::string binFileFormat("BinOcaf");
#endif


#if OCC_VERSION_HEX < 0x070000
static std::string modelFileFilter("EasyModeling Files (*.std *.xml)");
#else
static std::string modelFileFilter("EasyModeling Files (*.cbf *.xml)");
#endif



OCAFDocumentCtrl::
OCAFDocumentCtrl()
{
  m_OCAFApp = new OCAF_Application;

#if OCC_VERSION_HEX < 0x070000

#else
  BinDrivers::DefineFormat(m_OCAFApp);
  XmlDrivers::DefineFormat(m_OCAFApp);
  try{
    UnitsAPI::SetLocalSystem(UnitsAPI_MDTV);
  }catch (Standard_Failure) {
    cerr<<"Fatal Error in units initialisation"<<endl;
  }
#endif 

}


OCAFDocumentCtrl::
~OCAFDocumentCtrl()
{
  if(!m_OCAFApp.IsNull()){
    m_OCAFApp.Nullify();
  }

  if(!m_OCAFDoc.IsNull()){
    m_OCAFDoc.Nullify();
  }
}


void
OCAFDocumentCtrl::
OnNewDocument()
{
  if(!m_OCAFApp.IsNull()){
    m_OCAFApp->NewDocument(binFileFormat.c_str(), m_OCAFDoc);
  }
}


Handle(OCAF_Application)  
OCAFDocumentCtrl::
GetOCAFApplication() 
{ 
  return m_OCAFApp; 
}


Handle(TDocStd_Document)     
OCAFDocumentCtrl::
GetOCAFDoc()          
{ 
  return m_OCAFDoc; 
}


Handle(TDataStd_TreeNode) 
OCAFDocumentCtrl::
GetRoot()
{
  Handle(TDataStd_TreeNode) aRoot;
  if(! m_OCAFDoc.IsNull() ){
    TDF_Label aRootLabel = m_OCAFDoc->GetData()->Root();
    aRootLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aRoot);
  }
  return aRoot;
}









bool
OCAFDocumentCtrl::
Extention(std::string& fn)
{
  bool rep = false;
  size_t num = fn.length();

  if(num>4){
    size_t pos = num-4;
    std::string tmpStr = fn.substr(pos, 4);

    if( (tmpStr == binFileTypeExtention) || (tmpStr == xmlFileTypeExtention) )
      rep = true;
  }
  return rep;
}


void  
OCAFDocumentCtrl::
OnSave()
{
  std::string fn;
  std::string resultName;

  if (Extention(resultName) == false){
    // default is saved as binary formal
    resultName = resultName + binFileTypeExtention;
  }

  if( OnSaveDocument( resultName ) ){;}
  else{;} 
}


//================================================================
// Function : MdiSubWindow::OnSaveDocument
// Purpose  : saving of document
//================================================================
Standard_Boolean 
OCAFDocumentCtrl::
OnSaveDocument(const std::string& SPath) 
{
  cout<<"SPath = "<<SPath<<endl;

  if(this->GetOCAFDoc().IsNull()) return Standard_False;
  TCollection_ExtendedString TPath(SPath.c_str());

  if (TPath.SearchFromEnd( xmlFileTypeExtention.c_str() ) > 0) {
    // The document must be saved in XML format
    this->GetOCAFDoc()->ChangeStorageFormat( xmlFileFormat.c_str() );
  } else if (TPath.SearchFromEnd( binFileTypeExtention.c_str() ) > 0) {
    // The document must be saved in binary format
    this->GetOCAFDoc()->ChangeStorageFormat( binFileFormat.c_str() );
  }

  try {
    this->GetOCAFApplication()->SaveAs(this->GetOCAFDoc(),TPath);
  }
  catch(...) {
    return Standard_False;
  }

  return Standard_True;
}

