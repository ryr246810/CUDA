#include<SI_SC_IntegralEquation.hxx>
#include<GridVertexData.hxx>


void Advance_OneElecElem_SI_SC_Damping_1(GridVertexData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex)
{
  Standard_Real newAEPhysData = 
    (1.0-damping_scale)*theData->GetSweptPhysData(dynEIndex) 
    + damping_scale*theData->GetSweptPhysData(AEIndex);

  theData->SetSweptPhysData(AEIndex, newAEPhysData);
}


void Advance_OneElecElem_SI_SC_Damping_2(GridVertexData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer BEIndex,
					 Standard_Real preElecPhysData)
{
  Standard_Real result = 
    (1.0+0.5*damping_scale)*theData->GetSweptPhysData(dynEIndex) 
    - 0.5*preElecPhysData
    + 0.5*(1-damping_scale)*theData->GetSweptPhysData(AEIndex);

  theData->SetSweptPhysData(BEIndex, result);
}


void Advance_OneElecElem_SI_SC_Damping_3(GridVertexData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer PREIndex)
{
  // E'(n-2) = E(n-2) + theta*E'(n-3)
  Standard_Real newAEPhysData = 
    theData->GetSweptPhysData(PREIndex) 
    + damping_scale*theData->GetSweptPhysData(AEIndex);
  theData->SetSweptPhysData(AEIndex, newAEPhysData);

  Standard_Real newPREPhysData = theData->GetSweptPhysData(dynEIndex);
  theData->SetSweptPhysData(PREIndex, newPREPhysData);
}


void Advance_OneElecElem_SI_SC_Damping_4(GridVertexData* theData, 
					 Standard_Integer dynEIndex,
					 Standard_Real damping_scale,
					 Standard_Integer AEIndex,
					 Standard_Integer BEIndex,
					 Standard_Integer PREIndex)
{
  //(1+0.5*theta)*E(n)-theta*(1-0.5*theta)E(n-1)+0.5*(1.0-theta)*(1.0-theta)*theta*E'(n-2)
  Standard_Real result = 
    (1.0+0.5*damping_scale)*theData->GetSweptPhysData(dynEIndex) 
    - damping_scale*(1-0.5*damping_scale)*theData->GetSweptPhysData(PREIndex)
    + 0.5*(1-damping_scale)*(1-damping_scale)*damping_scale*theData->GetSweptPhysData(AEIndex);

  theData->SetSweptPhysData(BEIndex, result);
}
