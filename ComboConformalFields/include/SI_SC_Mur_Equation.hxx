#ifndef _SI_SC_Mur_Equation_HeaderFile
#define _SI_SC_Mur_Equation_HeaderFile


#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <BaseFunctionDefine.hxx>
#include <TFunc.hxx>

void Advance_Mur_OneElecElem_SI_SC(GridEdgeData* theData, GridEdgeData* theAssistantData, 
				   Standard_Real Dt,
				   Standard_Real si_scale,
				   Standard_Real  theVBar,
				   Standard_Integer dynEFldIndx,
				   Standard_Integer preTStepEFldIndx);

void Advance_Mur_OneElecElem_SI_SC_Damping(GridEdgeData* theData, GridEdgeData* theAssistantData, 
					   Standard_Real Dt, Standard_Real si_scale,
					   Standard_Real theVBar,
					   Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
					   Standard_Real damping_scale,
					   Standard_Integer EIndex, Standard_Integer BEIndex);

void Advance_Mur_OneElecElem_SI_SC_Damping_new(GridEdgeData* theData, GridEdgeData* theAssistantData, 
					       Standard_Real Dt, Standard_Real si_scale,
					       Standard_Real theVBar,
					       Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
					       Standard_Real damping_scale,
					       Standard_Integer EIndex, Standard_Integer BEIndex,
					       Standard_Integer thePREIndex);

void Advance_Mur_OneElecElem_SI_SC_TFunc(GridEdgeData* theData, GridEdgeData* theAssistantData, 
					 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
					 Standard_Real  theVBar,
					 Standard_Integer dynEFldIndx,  Standard_Integer preTStepEFldIndx,
					 Standard_Real amp, TFunc* tfuncPtr);

void Advance_Mur_OneElecElem_SI_SC_TETM01Func(GridEdgeData* theData, GridEdgeData* theAssistantData, 
					 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
					 Standard_Real  theVBar,
					 Standard_Integer dynEFldIndx,  Standard_Integer preTStepEFldIndx,
					 Standard_Real amp, TFunc* tfuncPtr, double frequence);

void Advance_Mur_OneElecElem_SI_SC_TFunc_Damping(GridEdgeData* theData, GridEdgeData* theAssistantData, 
						 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						 Standard_Real  theVBar,
						 Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						 Standard_Real amp, TFunc* tfuncPtr,
						 Standard_Real damping_scale,
						 Standard_Integer theAEIndex, Standard_Integer theBEIndex);

void Advance_Mur_OneElecElem_SI_SC_TETM01Func_Damping(GridEdgeData* theData, GridEdgeData* theAssistantData, 
						 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						 Standard_Real  theVBar,
						 Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						 Standard_Real amp, TFunc* tfuncPtr,
						 Standard_Real damping_scale,
						 Standard_Integer theAEIndex, Standard_Integer theBEIndex, double frequence);

void Advance_Mur_OneElecElem_SI_SC_TFunc_Damping_new(GridEdgeData* theData, GridEdgeData* theAssistantData, 
						     Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						     Standard_Real  theVBar,
						     Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						     Standard_Real amp, TFunc* tfuncPtr,
						     Standard_Real damping_scale,
						     Standard_Integer theAEIndex, Standard_Integer theBEIndex,
						     Standard_Integer thePREIndex);



/**************************************************************************************************/

void Advance_Mur_OneElecElem_SI_SC(GridVertexData* theData, GridVertexData* theAssistantData, 
				   Standard_Real Dt,
				   Standard_Real si_scale,
				   Standard_Real  theVBar,
				   Standard_Integer dynEFldIndx,
				   Standard_Integer preTStepEFldIndx);

void Advance_Mur_OneElecElem_SI_SC_Damping(GridVertexData* theData, GridVertexData* theAssistantData, 
					   Standard_Real Dt, Standard_Real si_scale,
					   Standard_Real theVBar,
					   Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
					   Standard_Real damping_scale,
					   Standard_Integer EIndex, Standard_Integer BEIndex);

void Advance_Mur_OneElecElem_SI_SC_Damping_new(GridVertexData* theData, GridVertexData* theAssistantData, 
					       Standard_Real Dt, Standard_Real si_scale,
					       Standard_Real theVBar,
					       Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
					       Standard_Real damping_scale,
					       Standard_Integer EIndex, Standard_Integer BEIndex,
					       Standard_Integer thePREIndex);

void Advance_Mur_OneElecElem_SI_SC_TFunc(GridVertexData* theData, GridVertexData* theAssistantData, 
					 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
					 Standard_Real  theVBar,
					 Standard_Integer dynEFldIndx,  Standard_Integer preTStepEFldIndx,
					 Standard_Real amp, TFunc* tfuncPtr);

void Advance_Mur_OneElecElem_SI_SC_TETM01Func(GridVertexData* theData, GridVertexData* theAssistantData, 
					 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
					 Standard_Real  theVBar,
					 Standard_Integer dynEFldIndx,  Standard_Integer preTStepEFldIndx,
					 Standard_Real amp, TFunc* tfuncPtr, double frequence);

void Advance_Mur_OneElecElem_SI_SC_TFunc_Damping(GridVertexData* theData, GridVertexData* theAssistantData, 
						 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						 Standard_Real  theVBar,
						 Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						 Standard_Real amp, TFunc* tfuncPtr,
						 Standard_Real damping_scale,
						 Standard_Integer theAEIndex, Standard_Integer theBEIndex);

void Advance_Mur_OneElecElem_SI_SC_TETM01Func_Damping(GridVertexData* theData, GridVertexData* theAssistantData, 
						 Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						 Standard_Real  theVBar,
						 Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						 Standard_Real amp, TFunc* tfuncPtr,
						 Standard_Real damping_scale,
						 Standard_Integer theAEIndex, Standard_Integer theBEIndex, double frequence);

void Advance_Mur_OneElecElem_SI_SC_TFunc_Damping_new(GridVertexData* theData, GridVertexData* theAssistantData, 
						     Standard_Real t, Standard_Real Dt, Standard_Real si_scale,
						     Standard_Real  theVBar,
						     Standard_Integer dynEFldIndx, Standard_Integer preTStepEFldIndx,
						     Standard_Real amp, TFunc* tfuncPtr,
						     Standard_Real damping_scale,
						     Standard_Integer theAEIndex, Standard_Integer theBEIndex,
						     Standard_Integer thePREIndex);
#endif
