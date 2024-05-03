#include <SI_SC_CPML_Equation.hxx>
#include <SI_SC_IntegralEquation.hxx>
// #include "omp.h"

/****************************************************************************************/
void Compute_DualContourValue1_SI_SC(GridVertexData* theData,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue)
{
  DualContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetSharedTDFaces();
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
    GridEdgeData* currEdgeData = (GridEdgeData*)theOutLineTElems[index].GetData();
    Standard_Integer currEdgeDir = currEdgeData->GetDir();
    Standard_Integer currRelativeDir = theOutLineTElems[index].GetRelatedDir();

    DualContourValue += 
      currEdgeData->GetSweptPhysData(theMagPhysIndex) * 
      currEdgeData->GetDualSweptGeomDim() * 
      currRelativeDir *
      (TwoDim_DirBump(currEdgeDir,1) - ThreeDim_DirBump(2,1)) / (ThreeDim_DirBump(2,2) - ThreeDim_DirBump(2,1)) ;
  }
};


void Compute_DualContourValue2_SI_SC(GridVertexData* theData, 
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue)
{
  DualContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetSharedTDFaces();
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
    GridEdgeData* currEdgeData = (GridEdgeData*)theOutLineTElems[index].GetData();
    Standard_Integer currEdgeDir = currEdgeData->GetDir();
    Standard_Integer currRelativeDir = theOutLineTElems[index].GetRelatedDir();

    DualContourValue += 
      currEdgeData->GetSweptPhysData(theMagPhysIndex) *
      currEdgeData->GetDualSweptGeomDim() * 
      currRelativeDir *
      (TwoDim_DirBump(currEdgeDir,1) - ThreeDim_DirBump(2,2)) / (ThreeDim_DirBump(2,1) - ThreeDim_DirBump(2,2)) ;
  }
};



/****************************************************************************************/
void Compute_PE1_SI_SC(GridVertexData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE1Index,
		       Standard_Real DualContourValue)
{
  Standard_Integer theTruncDir = ThreeDim_DirBump(2,1);

  Standard_Real a = 0.0;
  Standard_Real b = 0.0;
  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * theData->GetSweptPhysData(thePE1Index) + 
    a * DualContourValue / theData->GetDualSweptGeomDim();

  theData->SetSweptPhysData(thePE1Index, result);
};



void Compute_PE2_SI_SC(GridVertexData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE2Index,
		       Standard_Real DualContourValue)
{
  Standard_Integer theTruncDir = ThreeDim_DirBump(2,2);

  Standard_Real a = 0.0;
  Standard_Real b = 0.0;
  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * theData->GetSweptPhysData(thePE2Index) + 
    a * DualContourValue / theData->GetDualSweptGeomDim();

  theData->SetSweptPhysData(thePE2Index, result);
};



/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC(GridVertexData* theData,
				    Standard_Real Dt, 
				    Standard_Real si_scale,
				    Standard_Integer theElecPhysIndex,
				    Standard_Integer thePE1Index,
				    Standard_Integer thePE2Index,
				    Standard_Integer theMagPhysIndex)
{
  Standard_Real dt = si_scale*Dt;

  Standard_Real DualContourValue1;
  Compute_DualContourValue1_SI_SC(theData, theMagPhysIndex, DualContourValue1);
  Compute_PE1_SI_SC(theData, dt, thePE1Index, DualContourValue1);

  Standard_Real DualContourValue2;
  Compute_DualContourValue2_SI_SC(theData, theMagPhysIndex, DualContourValue2);
  Compute_PE2_SI_SC(theData, dt, thePE2Index, DualContourValue2);  

  Standard_Real C0 = 
    (theData->GetEpsilon() - 0.5*theData->GetSigma()*dt)/
    (theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);
  Standard_Real C2 = dt/(theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);

  Standard_Integer Dir1 = ThreeDim_DirBump(2,1);
  Standard_Real Kappa1 = theData->GetPMLKappa(Dir1);

  Standard_Integer Dir2 = ThreeDim_DirBump(2,2);
  Standard_Real Kappa2 = theData->GetPMLKappa(Dir2);

  Standard_Real result =  
    C0 * theData->GetSweptPhysData(theElecPhysIndex) 
    + C2 * (DualContourValue1/Kappa1 + DualContourValue2/Kappa2)/(theData->GetDualSweptGeomDim())
    + C2 * theData->GetSweptPhysData(thePE1Index)
    + C2 * theData->GetSweptPhysData(thePE2Index);

  theData->SetSweptPhysData(theElecPhysIndex, result);
};



void Advance_CPML_ElecElems_SI_SC(vector<GridVertexData*>& theDatas,
				  Standard_Real Dt, 
				  Standard_Real si_scale,
				  Standard_Integer theElecPhysIndex,
				  Standard_Integer thePE1Index,
				  Standard_Integer thePE2Index,
				  Standard_Integer theMagPhysIndex)
{
  Standard_Size nbe = theDatas.size();
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_CPML_OneElecElem_SI_SC(theDatas[eindex], Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);
  }
}



/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC_Damping(GridVertexData* theData,
					    Standard_Real Dt, 
					    Standard_Real si_scale,
					    Standard_Integer theElecPhysIndex,
					    Standard_Integer thePE1Index,
					    Standard_Integer thePE2Index,
					    Standard_Integer theMagPhysIndex,
					    Standard_Real damping_scale,
					    Standard_Integer theAEIndex,
					    Standard_Integer theBEIndex)
{
  Advance_OneElecElem_SI_SC_Damping_1(theData, theElecPhysIndex, damping_scale, theAEIndex);

  Standard_Real preElecPhysData = theData->GetSweptPhysData(theElecPhysIndex);
  Advance_CPML_OneElecElem_SI_SC(theData, Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_Damping_2(theData, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, preElecPhysData);
};


void Advance_CPML_ElecElems_SI_SC_Damping(vector<GridVertexData*>& theDatas,
					  Standard_Real Dt, 
					  Standard_Real si_scale,
					  Standard_Integer theElecPhysIndex,
					  Standard_Integer thePE1Index,
					  Standard_Integer thePE2Index,
					  Standard_Integer theMagPhysIndex,
					  Standard_Real damping_scale,
					  Standard_Integer theAEIndex,
					  Standard_Integer theBEIndex)
{
  Standard_Size nbe = theDatas.size();
  //#pragma omp parallel for
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_CPML_OneElecElem_SI_SC_Damping(theDatas[eindex],
					   Dt, si_scale,
					   theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex,
					   damping_scale, theAEIndex, theBEIndex);
  }
}





/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC_Damping_new(GridVertexData* theData,
						Standard_Real Dt, 
						Standard_Real si_scale,
						Standard_Integer theElecPhysIndex,
						Standard_Integer thePE1Index,
						Standard_Integer thePE2Index,
						Standard_Integer theMagPhysIndex,
						Standard_Real damping_scale,
						Standard_Integer theAEIndex,
						Standard_Integer theBEIndex,
						Standard_Integer thePREIndex)
{
  Advance_OneElecElem_SI_SC_Damping_3(theData, theElecPhysIndex, damping_scale, theAEIndex, thePREIndex);

  Advance_CPML_OneElecElem_SI_SC(theData, Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_Damping_4(theData, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, thePREIndex);
};



void Advance_CPML_ElecElems_SI_SC_Damping_new(vector<GridVertexData*>& theDatas,
					      Standard_Real Dt, 
					      Standard_Real si_scale,
					      Standard_Integer theElecPhysIndex,
					      Standard_Integer thePE1Index,
					      Standard_Integer thePE2Index,
					      Standard_Integer theMagPhysIndex,
					      Standard_Real damping_scale,
					      Standard_Integer theAEIndex,
					      Standard_Integer theBEIndex,
					      Standard_Integer thePREIndex)
{
  Standard_Size nbe = theDatas.size();
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_CPML_OneElecElem_SI_SC_Damping_new(theDatas[eindex],
					       Dt, si_scale,
					       theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex,
					       damping_scale, theAEIndex, theBEIndex, thePREIndex);
  }
}
