#include <TxMakerMap.h>
#include <FieldsDgnSets.hxx>
#include <FieldsDgnBase.hxx>


FieldsDgnSets::
FieldsDgnSets()
{
};


FieldsDgnSets::
FieldsDgnSets(const FieldsDefineCntr* thecntr)
{
  Init(thecntr);
}


void 
FieldsDgnSets::
Init(const FieldsDefineCntr* thecntr)
{
  m_FldsDefCntr = thecntr;
}


const FieldsDefineCntr*
FieldsDgnSets::
GetFldsDefCntr() const 
{
  return m_FldsDefCntr;
}


void 
FieldsDgnSets::
Append(FieldsDgnBase* _oneDgn)
{
  m_DgnGrp.push_back(_oneDgn);
};


FieldsDgnSets::
~FieldsDgnSets()
{
  for (Standard_Size idx = m_DgnGrp.size(); idx>0; delete m_DgnGrp[--idx]);
  m_DgnGrp.clear();
};


void 
FieldsDgnSets::
DumpHead(std::ostream& out)
{
  out << "time" ;
  for (Standard_Size idx = 0; idx<m_DgnGrp.size(); idx++){
    out<<"\t\t"<< (m_DgnGrp[idx]->GetName());
  }
  out<<std::endl;
}


void
FieldsDgnSets::
Dump(std::ostream& out)
{
  out<< GetCurTime();
  for (Standard_Size idx = 0; idx<m_DgnGrp.size(); idx++){
    out<< "\t\t" << m_DgnGrp[idx]->GetValue();
  }
  out<<std::endl;
}


void 
FieldsDgnSets::
Advance()
{
  for (Standard_Size idx = 0; idx<m_DgnGrp.size(); idx++){
    m_DgnGrp[idx]->Advance();
  }
  DynObj::Advance();
}



void 
FieldsDgnSets::
SetAttrib(const TxHierAttribSet& tha)
{
  std::vector< std::string > fldDgnNames = tha.getNamesOfType("FieldsDgn");

  if( fldDgnNames.size() ){
    std::cout << "\t FieldsDgnSets::SetAttrib-----FieldsDgns are:";
    for(size_t i=0; i<fldDgnNames.size(); ++i)
      std::cout << " " << fldDgnNames[i];
    std::cout << std::endl;
  }

  // Add in all the ports
  for(size_t i=0; i<fldDgnNames.size(); ++i){
    TxHierAttribSet attribs = tha.getAttrib(fldDgnNames[i]);
    if(attribs.hasString("kind")) {
      std::string kind = attribs.getString("kind");
      FieldsDgnBase* oneNewDgn= TxMakerMap<FieldsDgnBase>::getNew(kind);

      if(oneNewDgn == 0){
	std::cout << "\t Dgn of kind " << kind << " not found." << std::endl;
	continue;
      }
      oneNewDgn->Init(m_FldsDefCntr);
      oneNewDgn->SetDelTime(this->GetDelTime());
      oneNewDgn->SetAttrib(attribs);

      this->Append(oneNewDgn);
    }
  }
}

