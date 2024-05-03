#ifndef _OCAFDocumentCtrl_HeaderFile
#define _OCAFDocumentCtrl_HeaderFile

#include <OCAF_Application.hxx>
#include <TDocStd_Document.hxx>

#include <TDataStd_TreeNode.hxx>

#include <TxHierAttribSet.h>

#include <string>

class OCAFDocumentCtrl
{
public:
  OCAFDocumentCtrl();
  ~OCAFDocumentCtrl();


public:
  Handle(OCAF_Application)  GetOCAFApplication();
  Handle(TDocStd_Document)  GetOCAFDoc();
  Handle(TDataStd_TreeNode) GetRoot();

  void OnNewDocument();


public:
  bool Extention(std::string& fn);
  void OnSave();

  Standard_Boolean OnSaveDocument(const std::string& SPath);
  void OnCloseDocument();
  void ReadAttrib(TxHierAttribSet& tas);


private:
  std::string m_Name;
  Handle(OCAF_Application) m_OCAFApp;
  Handle(TDocStd_Document) m_OCAFDoc;
}; 

#endif
