
#include <SI_SC_Mur_Equation.hxx>
#include <SI_SC_IntegralEquation.hxx>
#include <GridVertexData.hxx>

//--------------------------------------------------------->>>
void Advance_Mur_OneElecElem_SI_SC(GridVertexData* theData, 
				   GridVertexData* theAssistantData, 
				   Standard_Real Dt, 
				   Standard_Real si_scale,
				   Standard_Real  theVBar,
				   Standard_Integer dynEIndex,
				   Standard_Integer preTStepEFldIndx)
{
  Standard_Real dt = si_scale*Dt;
  Standard_Real vbar = theVBar*dt;

  Standard_Real currE =  theAssistantData->GetSweptPhysData(dynEIndex);
  Standard_Real oldE = theData->GetSweptPhysData(preTStepEFldIndx);
  
  Standard_Real tmpE = (oldE - currE*(1.0-vbar))/(1.0+vbar);
  
  oldE = tmpE*(1.0-vbar) + currE*(1.0+vbar); 
  theData->SetSweptPhysData(preTStepEFldIndx, oldE);
  theData->SetSweptPhysData(dynEIndex, tmpE);
};


void Advance_Mur_OneElecElem_SI_SC_Damping(GridVertexData* theData, 
					   GridVertexData* theAssistantData, 
					   Standard_Real Dt, 
					   Standard_Real si_scale,
					   Standard_Real  theVBar,
					   Standard_Integer dynEIndex,
					   Standard_Integer preTStepEFldIndx,
					   Standard_Real damping_scale,
					   Standard_Integer AEIndex,
					   Standard_Integer BEIndex)
{
  Advance_OneElecElem_SI_SC_Damping_1(theData,dynEIndex, damping_scale, AEIndex);

  Standard_Real preEdgePhysData = theData->GetSweptPhysData(dynEIndex);
  Advance_Mur_OneElecElem_SI_SC(theData, theAssistantData, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx);

  Advance_OneElecElem_SI_SC_Damping_2(theData, dynEIndex, damping_scale, AEIndex, BEIndex, preEdgePhysData);
};







void Advance_Mur_OneElecElem_SI_SC_Damping_new(GridVertexData* theData, 
					       GridVertexData* theAssistantData, 
					       Standard_Real Dt, 
					       Standard_Real si_scale,
					       Standard_Real  theVBar,
					       Standard_Integer dynEIndex,
					       Standard_Integer preTStepEFldIndx,
					       Standard_Real damping_scale,
					       Standard_Integer AEIndex,
					       Standard_Integer BEIndex,
					       Standard_Integer PREIndex)
{
  Advance_OneElecElem_SI_SC_Damping_3(theData,dynEIndex, damping_scale, AEIndex, PREIndex);
  Advance_Mur_OneElecElem_SI_SC(theData, theAssistantData, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx);
  Advance_OneElecElem_SI_SC_Damping_4(theData, dynEIndex, damping_scale, AEIndex, BEIndex, PREIndex);
};






void 
Advance_Mur_OneElecElem_SI_SC_TFunc(GridVertexData* theData, 
				    GridVertexData* theAssistantData, 
				    Standard_Real t, 
				    Standard_Real Dt, 
				    Standard_Real si_scale,
				    Standard_Real  theVBar,
				    Standard_Integer dynEIndex,
				    Standard_Integer preTStepEFldIndx,
				    Standard_Real amp,
				    TFunc* tfuncPtr)
{
  Standard_Real dt =si_scale*Dt;
  Standard_Real vbar = theVBar*dt;

  Standard_Real currE =  theAssistantData->GetSweptPhysData(dynEIndex);
  Standard_Real oldE = theData->GetSweptPhysData(preTStepEFldIndx);
  
  Standard_Real Ebar = amp*tfuncPtr->operator()(t+dt);
  Standard_Real Ebar2 = amp*tfuncPtr->operator()(t+dt-dt/vbar);
  
  Standard_Real tmpE = (oldE - (currE-Ebar2)*(1.0-vbar))/(1.0+vbar) + Ebar;
  oldE = (tmpE - Ebar)*(1.0-vbar) + (currE-Ebar2)*(1.0+vbar); 
  
  theData->SetSweptPhysData(preTStepEFldIndx, oldE);
  theData->SetSweptPhysData(dynEIndex, tmpE);
}


void 
Advance_Mur_OneElecElem_SI_SC_TFunc_Damping(GridVertexData* theData, 
					    GridVertexData* theAssistantData, 
					    Standard_Real t, 
					    Standard_Real Dt, 
					    Standard_Real si_scale,
					    Standard_Real  theVBar,
					    Standard_Integer dynEIndex,
					    Standard_Integer preTStepEFldIndx,
					    Standard_Real amp,
					    TFunc* tfuncPtr,
					    Standard_Real damping_scale,
					    Standard_Integer AEIndex,
					    Standard_Integer BEIndex)
{
  Advance_OneElecElem_SI_SC_Damping_1(theData,dynEIndex, damping_scale, AEIndex);

  Standard_Real preEdgePhysData = theData->GetSweptPhysData(dynEIndex);
  Advance_Mur_OneElecElem_SI_SC_TFunc(theData, theAssistantData, t, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx, amp, tfuncPtr);
  
  Advance_OneElecElem_SI_SC_Damping_2(theData, dynEIndex, damping_scale, AEIndex, BEIndex, preEdgePhysData);
}


void 
Advance_Mur_OneElecElem_SI_SC_TFunc_Damping_new(GridVertexData* theData, 
						GridVertexData* theAssistantData, 
						Standard_Real t, 
						Standard_Real Dt, 
						Standard_Real si_scale,
						Standard_Real  theVBar,
						Standard_Integer dynEIndex,
						Standard_Integer preTStepEFldIndx,
						Standard_Real amp,
						TFunc* tfuncPtr,
						Standard_Real damping_scale,
						Standard_Integer AEIndex,
						Standard_Integer BEIndex,
						Standard_Integer PREIndex)
{
  Advance_OneElecElem_SI_SC_Damping_3(theData,dynEIndex, damping_scale, AEIndex, PREIndex);
  Advance_Mur_OneElecElem_SI_SC_TFunc(theData, theAssistantData, t, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx, amp, tfuncPtr);
  Advance_OneElecElem_SI_SC_Damping_4(theData, dynEIndex, damping_scale, AEIndex, BEIndex, PREIndex);
}
