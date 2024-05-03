#include<SI_SC_IntegralEquation.hxx>
#include<GridEdge.hxx>


void Advance_OneElecElem_SI_SC_AlongAxis_Damping_1(vector<GridEdgeData*> theDatas,
					 Standard_Size m_PhiNumber,
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex)
{
 for(Standard_Size i=0; i<m_PhiNumber; i++){

  Standard_Real newAEPhysData = 
    (1.0-damping_scale)*theDatas[i]->GetPhysData(dynEIndex) 
    + damping_scale*theDatas[i]->GetPhysData(AEIndex);
  
  theDatas[i]->SetPhysData(AEIndex, newAEPhysData);
  }
}


void Advance_OneElecElem_SI_SC_AlongAxis_Damping_2(vector<GridEdgeData*> theDatas,
					 Standard_Size m_PhiNumber,
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer BEIndex,
					 vector<Standard_Real> preElecPhysData)
{

 for(Standard_Size i=0; i<m_PhiNumber; i++){
  Standard_Real result = 
    (1.0+0.5*damping_scale)*theDatas[i]->GetPhysData(dynEIndex) 
    - 0.5*preElecPhysData[i]
    + 0.5*(1-damping_scale)*theDatas[i]->GetPhysData(AEIndex);

  theDatas[i]->SetPhysData(BEIndex, result);
  }
}



