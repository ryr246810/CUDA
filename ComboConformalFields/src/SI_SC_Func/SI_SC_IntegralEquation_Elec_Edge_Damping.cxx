#include<SI_SC_IntegralEquation.hxx>
#include<GridEdge.hxx>


void Advance_OneElecElem_SI_SC_Damping_1(GridEdgeData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex)
{
  Standard_Real newAEPhysData = 
    (1.0-damping_scale)*theData->GetPhysData(dynEIndex) 
    + damping_scale*theData->GetPhysData(AEIndex);
  
  theData->SetPhysData(AEIndex, newAEPhysData);
}


void Advance_OneElecElem_SI_SC_Damping_2(GridEdgeData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer BEIndex,
					 Standard_Real preElecPhysData)
{
  Standard_Real result = 
    (1.0+0.5*damping_scale)*theData->GetPhysData(dynEIndex) 
    - 0.5*preElecPhysData
    + 0.5*(1-damping_scale)*theData->GetPhysData(AEIndex);

  theData->SetPhysData(BEIndex, result);
}


void Advance_OneElecElem_SI_SC_Damping_3(GridEdgeData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer PREIndex)
{
  // E'(n-2) = E(n-2) + theta*E'(n-3)
  Standard_Real newAEPhysData = 
    theData->GetPhysData(PREIndex) 
    + damping_scale*theData->GetPhysData(AEIndex);
  theData->SetPhysData(AEIndex, newAEPhysData);

  Standard_Real newPREPhysData = theData->GetPhysData(dynEIndex);
  theData->SetPhysData(PREIndex, newPREPhysData);
}


void Advance_OneElecElem_SI_SC_Damping_4(GridEdgeData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer BEIndex,
					 Standard_Integer PREIndex)
{
  //(1+0.5*theta)*E(n)-theta*(1-0.5*theta)E(n-1)+0.5*(1.0-theta)*(1.0-theta)*theta*E'(n-2)
  Standard_Real result = 
    (1.0+0.5*damping_scale)*theData->GetPhysData(dynEIndex) 
    - damping_scale*(1-0.5*damping_scale)*theData->GetPhysData(PREIndex)
    + 0.5*(1-damping_scale)*(1-damping_scale)*damping_scale*theData->GetPhysData(AEIndex);

  theData->SetPhysData(BEIndex, result);
}
