
#include <SI_SC_Mur_ElecFieldsSet.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>

#include <PortDataFunc.hxx>

SI_SC_Mur_ElecFieldsSet::
SI_SC_Mur_ElecFieldsSet()
  :FieldsBase()
{
}


SI_SC_Mur_ElecFieldsSet::
SI_SC_Mur_ElecFieldsSet(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr, INCLUDING)
{

}


SI_SC_Mur_ElecFieldsSet::
~SI_SC_Mur_ElecFieldsSet()
{
  Clear();
}


void
SI_SC_Mur_ElecFieldsSet::
Clear()
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts.begin(); iter!=m_MurPorts.end(); ++iter){
    SI_SC_Mur_ElecFields* tmpFlds = *iter;
    *iter = NULL;
    delete tmpFlds;
  }
  m_MurPorts.clear();
}


void 
SI_SC_Mur_ElecFieldsSet::
Advance()
{
  DynObj::Advance();
}


void 
SI_SC_Mur_ElecFieldsSet::
Advance_SI(const Standard_Real si_scale)
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts.begin(); iter!=m_MurPorts.end(); ++iter){
    (*iter)->Advance_SI( si_scale);
  }
}


void 
SI_SC_Mur_ElecFieldsSet::
Advance_SI_Damping(const Standard_Real si_scale, 
		   const Standard_Real damping_scale)
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts.begin(); iter!=m_MurPorts.end(); ++iter){
    (*iter)->Advance_SI_Damping( si_scale, damping_scale);
  }
}


void 
SI_SC_Mur_ElecFieldsSet::
ZeroPhysDatas()
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts.begin(); iter!=m_MurPorts.end(); ++iter){
    (*iter)->ZeroPhysDatas();
  }
}


void 
SI_SC_Mur_ElecFieldsSet::
Setup()
{
  Standard_Real dt = this->GetDelTime();

  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = this->GetGridBndDatas()->GetPorts();
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;
  for(iter = thePorts->begin(); iter!=thePorts->end(); iter++){
    PortData currPort = iter->second;
    if(IsOpenMurPortType(currPort.m_Type)){
      SI_SC_Mur_ElecFields* oneNewMurPort = new SI_SC_Mur_ElecFields(m_FldsDefCntr, currPort);
      oneNewMurPort->SetDelTime(dt);
      oneNewMurPort->Setup();
      m_MurPorts.push_back(oneNewMurPort);
    }
  }
}
