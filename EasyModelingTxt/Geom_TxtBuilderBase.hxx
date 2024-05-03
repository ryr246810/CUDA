
#ifndef _Geom_TxtBuilderBase_HeaderFile
#define _Geom_TxtBuilderBase_HeaderFile

// std
#include <string>
#include <map>

// ocaf
#include <OCAFDocumentCtrl.hxx>
#include <TDF_AttributeMap.hxx>

// txbase
#include <TxHierAttribSet.h>

using namespace std;

class Geom_TxtBuilderBase
{
public:
  Geom_TxtBuilderBase();
  virtual ~Geom_TxtBuilderBase();

  virtual void Init(OCAFDocumentCtrl* _ocafDocCtrl);
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();

public:
  void MakeAvail(std::string name, Handle(TDataStd_TreeNode) data);

  static Handle(TDataStd_TreeNode) GetTreeNode(string name) {
    Handle(TDataStd_TreeNode) aNode;
    std::map<std::string, Handle(TDataStd_TreeNode), std::less<std::string> >::iterator iter = m_Objects.find(name);
    if(iter != m_Objects.end()) aNode = iter->second;
    return aNode;
  }


protected:
  OCAFDocumentCtrl* m_DocCtrl;
  Handle(TDataStd_TreeNode) m_Node;
  string m_Name;

  Standard_Boolean m_HasMaskDefined;
  Standard_Integer m_Mask;

  Standard_Boolean m_HasMaterialDefined;
  Standard_Integer m_MaterialIndex;

  static std::map<std::string, Handle(TDataStd_TreeNode), std::less<std::string> > m_Objects;
};

#endif

