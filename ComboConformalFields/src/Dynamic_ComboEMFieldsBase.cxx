#include <Dynamic_ComboEMFieldsBase.hxx>
#include <ComboFields_Dynamic_Srcs.hxx>
#include <ComboFields_Dynamic_Srcs_Cyl3D.hxx>

Dynamic_ComboEMFieldsBase::
Dynamic_ComboEMFieldsBase() 
  :  FieldsDefineBase()
{
  m_FldSrcs = NULL;
}


Dynamic_ComboEMFieldsBase::
Dynamic_ComboEMFieldsBase( const FieldsDefineCntr* _cntr) 
  :  FieldsDefineBase(_cntr)
{
  m_FldSrcs = NULL;
  m_FldSrcs_Cyl3D = NULL;
}


Dynamic_ComboEMFieldsBase::~Dynamic_ComboEMFieldsBase()
{
  if(m_FldSrcs != NULL) delete m_FldSrcs;
  if(m_FldSrcs_Cyl3D != NULL) delete m_FldSrcs_Cyl3D;
}




void
Dynamic_ComboEMFieldsBase::
InitFldSrcs()
{
  m_FldSrcs = new ComboFields_Dynamic_Srcs(this->GetFldsDefCntr());
  m_FldSrcs->SetDelTime(this->GetDelTime());
}

void
Dynamic_ComboEMFieldsBase::
InitFldSrcs_Cyl3D()
{
  m_FldSrcs_Cyl3D = new ComboFields_Dynamic_Srcs_Cyl3D(this->GetFldsDefCntr());
  m_FldSrcs_Cyl3D->SetDelTime(this->GetDelTime());
}


void
Dynamic_ComboEMFieldsBase::
SetFldSrcsAttrib(const string& theWorkDir,
		 const TxHierAttribSet& theFaceBndTha)
{
  m_FldSrcs->SetAttrib(theWorkDir, theFaceBndTha);
}

void
Dynamic_ComboEMFieldsBase::
SetFldSrcsAttrib_Cyl3D(const string& theWorkDir,
		 const TxHierAttribSet& theFaceBndTha)
{
  m_FldSrcs_Cyl3D->SetAttrib(theWorkDir, theFaceBndTha);
}


void 
Dynamic_ComboEMFieldsBase::
Setup()
{

}