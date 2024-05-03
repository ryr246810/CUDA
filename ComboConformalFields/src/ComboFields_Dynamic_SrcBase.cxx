#include <ComboFields_Dynamic_SrcBase.hxx>



ComboFields_Dynamic_SrcBase::
ComboFields_Dynamic_SrcBase()
  :FieldsSrcBase()
{
  m_WorkDir = "";
  m_PhiIndex = -1;
}


ComboFields_Dynamic_SrcBase::
ComboFields_Dynamic_SrcBase(const FieldsDefineCntr* theCntr, 
			    PhysDataDefineRule theRule)
  :FieldsSrcBase(theCntr, theRule)
{

  m_PhiIndex = -1;
}


ComboFields_Dynamic_SrcBase::
~ComboFields_Dynamic_SrcBase()
{
}


void 
ComboFields_Dynamic_SrcBase::
Setup()
{
	//cout<<"in ComboFields_Dynamic_SrcBase setup"<<endl;
}


void 
ComboFields_Dynamic_SrcBase::
SetWorkDir(const std::string theDir)
{
  m_WorkDir = theDir;
}
