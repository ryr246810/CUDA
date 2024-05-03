#include <Geom_TxtBuilderBase.hxx>
#include <OCAF_ColorMap.hxx>
#include <OCAF_Object.hxx>

std::map<std::string, Handle(TDataStd_TreeNode), std::less<std::string> > Geom_TxtBuilderBase::m_Objects;


void 
Geom_TxtBuilderBase::
MakeAvail(std::string name, Handle(TDataStd_TreeNode) data)
{
  m_Objects.insert( pair< string, Handle(TDataStd_TreeNode) >(name, data) );
}


Geom_TxtBuilderBase::
Geom_TxtBuilderBase()
{
  m_HasMaskDefined = Standard_False;
  m_Mask = 0;

  m_HasMaterialDefined = Standard_False;
  m_MaterialIndex = 0;
}


Geom_TxtBuilderBase::
~Geom_TxtBuilderBase()
{

}


void 
Geom_TxtBuilderBase::
Init(OCAFDocumentCtrl* _ocafDocCtrl)
{
  m_DocCtrl = _ocafDocCtrl;
}


void 
Geom_TxtBuilderBase::
InitVariable()
{

}


void 
Geom_TxtBuilderBase::
SetAttrib(const TxHierAttribSet& tha)
{
  if(tha.hasString("material")){
    string theMaterialName = tha.getString("material");
    m_MaterialIndex = OCAF_ColorMap::getMaterialIndex(theMaterialName, m_HasMaterialDefined);
    cout<<"m_MaterialIndex = "<<m_MaterialIndex<<endl;
  }else{
    m_HasMaterialDefined = Standard_False;
  }

  if(tha.hasOption("mask")){
    m_Mask = tha.getOption("mask");
    m_HasMaskDefined = Standard_True;
    cout<<"m_Mask = "<<m_Mask<<endl;
  }
}


void 
Geom_TxtBuilderBase::
Build()
{
  OCAF_Object anInterface(m_Node);
  anInterface.SetObjResultMaterial(m_MaterialIndex);
  anInterface.SetObjectMask(m_Mask);
}
