
#include <SI_SC_Mur_Equation.hxx>
#include <SI_SC_IntegralEquation.hxx>

//--------------------------------------------------------->>>
void Advance_Mur_OneElecElem_SI_SC(GridEdgeData* theData, 
				   GridEdgeData* theAssistantData, 
				   Standard_Real Dt, 
				   Standard_Real si_scale,
				   Standard_Real  theVBar,
				   Standard_Integer dynEIndex,
				   Standard_Integer preTStepEFldIndx)
{
  Standard_Real dt = si_scale*Dt;
  Standard_Real vbar = theVBar*dt;

  Standard_Real currE =  theAssistantData->GetPhysData(dynEIndex);
  Standard_Real oldE = theData->GetPhysData(preTStepEFldIndx);
  
  Standard_Real tmpE = (oldE - currE*(1.0-vbar))/(1.0+vbar);
  
  oldE = tmpE*(1.0-vbar) + currE*(1.0+vbar); 
  theData->SetPhysData(preTStepEFldIndx, oldE);
  theData->SetPhysData(dynEIndex, tmpE);
};


void Advance_Mur_OneElecElem_SI_SC_Damping(GridEdgeData* theData, 
					   GridEdgeData* theAssistantData, 
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
	// printf(" AE = %f###", theData->GetPhysData(AEIndex));

  Standard_Real preEdgePhysData = theData->GetPhysData(dynEIndex);
//   printf(" AE = %f###", preEdgePhysData);
	// printf(" AE = %f###", theVBar*(si_scale*Dt)); // 一致

  Advance_Mur_OneElecElem_SI_SC(theData, theAssistantData, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx);

  Advance_OneElecElem_SI_SC_Damping_2(theData, dynEIndex, damping_scale, AEIndex, BEIndex, preEdgePhysData);
};







void Advance_Mur_OneElecElem_SI_SC_Damping_new(GridEdgeData* theData, 
					       GridEdgeData* theAssistantData, 
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
Advance_Mur_OneElecElem_SI_SC_TFunc(GridEdgeData* theData, 
				    GridEdgeData* theAssistantData, 
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
	// printf("Ebar = %.20f \n", vbar);
  Standard_Real currE =  theAssistantData->GetPhysData(dynEIndex);
  
  Standard_Real oldE = theData->GetPhysData(preTStepEFldIndx);
  
  Standard_Real Ebar = amp*tfuncPtr->operator()(t+dt);
  Standard_Real Ebar2 = amp*tfuncPtr->operator()(t+dt-dt/vbar);

//   printf("Ebar2 = %.20f %.20f %.20f \n", amp, Ebar, Ebar2);
  
  Standard_Real tmpE = (oldE - (currE-Ebar2)*(1.0-vbar))/(1.0+vbar) + Ebar;
  
  oldE = (tmpE - Ebar)*(1.0-vbar) + (currE-Ebar2)*(1.0+vbar); 
  
  theData->SetPhysData(preTStepEFldIndx, oldE);
  theData->SetPhysData(dynEIndex, tmpE);

//   printf("\n \n m_MurPortVertexDatas_address = %.20f %.20f %.20f \n",  	amp, 
// 																		tfuncPtr->operator()(t+dt),
// 																		tfuncPtr->operator()(t+dt-dt/vbar));
}


void 
Advance_Mur_OneElecElem_SI_SC_TFunc_Damping(GridEdgeData* theData, 
					    GridEdgeData* theAssistantData, 
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

  Standard_Real preEdgePhysData = theData->GetPhysData(dynEIndex);
  Advance_Mur_OneElecElem_SI_SC_TFunc(theData, theAssistantData, t, Dt, si_scale, theVBar, dynEIndex, preTStepEFldIndx, amp, tfuncPtr);
  
  Advance_OneElecElem_SI_SC_Damping_2(theData, dynEIndex, damping_scale, AEIndex, BEIndex, preEdgePhysData);
}


void 
Advance_Mur_OneElecElem_SI_SC_TFunc_Damping_new(GridEdgeData* theData, 
						GridEdgeData* theAssistantData, 
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
