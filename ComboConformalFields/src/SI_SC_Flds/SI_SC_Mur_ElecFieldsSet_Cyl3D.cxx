
#include <SI_SC_Mur_ElecFieldsSet_Cyl3D.hxx>

#include<GridFaceData.cuh>
#include<GridEdgeData.hxx>
#include<GridFace.hxx>
#include<GridEdge.hxx>

#include <PortDataFunc.hxx>

SI_SC_Mur_ElecFieldsSet_Cyl3D::
SI_SC_Mur_ElecFieldsSet_Cyl3D()
  :FieldsBase()
{
}


SI_SC_Mur_ElecFieldsSet_Cyl3D::
SI_SC_Mur_ElecFieldsSet_Cyl3D(const FieldsDefineCntr* theCntr)
  :FieldsBase(theCntr, INCLUDING)
{

}


SI_SC_Mur_ElecFieldsSet_Cyl3D::
~SI_SC_Mur_ElecFieldsSet_Cyl3D()
{
  Clear();
}


void
SI_SC_Mur_ElecFieldsSet_Cyl3D::
Clear()
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts_Cyl3D.begin(); iter!=m_MurPorts_Cyl3D.end(); ++iter){
    SI_SC_Mur_ElecFields* tmpFlds = *iter;
    *iter = NULL;
    delete tmpFlds;
  }
  m_MurPorts_Cyl3D.clear();
}


void 
SI_SC_Mur_ElecFieldsSet_Cyl3D::
Advance()
{
  DynObj::Advance();
}


void 
SI_SC_Mur_ElecFieldsSet_Cyl3D::
Advance_SI(const Standard_Real si_scale)
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts_Cyl3D.begin(); iter!=m_MurPorts_Cyl3D.end(); ++iter){
    (*iter)->Advance_SI( si_scale);
  }
}


void 
SI_SC_Mur_ElecFieldsSet_Cyl3D::
Advance_SI_Damping(const Standard_Real si_scale, 
		   const Standard_Real damping_scale)
{
    // for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts_Cyl3D.begin(); iter!=m_MurPorts_Cyl3D.end(); ++iter){
    //     (*iter)->Advance_SI_Damping( si_scale, damping_scale);
    // }

    for(int i = 0; i < m_MurPorts_Cyl3D.size(); ++i){
      m_MurPorts_Cyl3D[i]->Advance_SI_Damping( si_scale, damping_scale);
    }
    // printf("m_MurPorts_Cyl3D.size() = %d \n", m_MurPorts_Cyl3D.size());
    // exit(-1);
    // printf("\n\n");
}


void 
SI_SC_Mur_ElecFieldsSet_Cyl3D::
ZeroPhysDatas()
{
  for(vector<SI_SC_Mur_ElecFields*>::iterator iter=m_MurPorts_Cyl3D.begin(); iter!=m_MurPorts_Cyl3D.end(); ++iter){
    (*iter)->ZeroPhysDatas();
  }
}


void 
SI_SC_Mur_ElecFieldsSet_Cyl3D::
Setup()
{
  Standard_Size m_DimPhi = GetGridGeom_Cyl3D()->GetDimPhi();
  Standard_Real dt = this->GetDelTime();

  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = this->GetGridBndDatas()->GetPorts();
  for(Standard_Size i =0 ;i < m_DimPhi ;i++){
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;
  for(iter = thePorts->begin(); iter!=thePorts->end(); iter++){
    PortData currPort = iter->second;
    if(IsOpenMurPortType(currPort.m_Type)){
      SI_SC_Mur_ElecFields* oneNewMurPort = new SI_SC_Mur_ElecFields(m_FldsDefCntr, currPort);
      oneNewMurPort->SetDelTime(dt);
      oneNewMurPort->SetPhiIndex(i);

      oneNewMurPort->Setup();
  
      m_MurPorts_Cyl3D.push_back(oneNewMurPort);
    }
  }
 }
}
