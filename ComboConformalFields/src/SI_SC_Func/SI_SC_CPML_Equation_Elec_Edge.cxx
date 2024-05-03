#include <SI_SC_CPML_Equation.hxx>
#include <SI_SC_IntegralEquation.hxx>
// #include "omp.h"

/****************************************************************************************/
void Compute_DualContourValue1_SI_SC(GridEdgeData* theData,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue)
{
  DualContourValue = 0;
  Standard_Integer nb = theData->GetSharedTFace().size();

  for(Standard_Integer index=0; index<nb; index ++){
    DualContourValue += 
      ( ((theData->GetSharedTFace()[index]).GetData())->GetPhysData(theMagPhysIndex) ) *
      ( ((theData->GetSharedTFace()[index]).GetData())->GetDualGeomDim() ) *
      (  (theData->GetSharedTFace()[index]).GetRelatedDir() ) ;
  }


};



/****************************************************************************************/
void Compute_PE1_SI_SC(GridEdgeData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE1Index,
		       Standard_Real DualContourValue)
{
  Standard_Integer theTruncDir = TwoDim_DirBump(theData->GetDir(),1);

  Standard_Real a = 0.0;
  Standard_Real b = 0.0;
  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * theData->GetPhysData(thePE1Index) + 
    a * DualContourValue / theData->GetDualGeomDim();

  theData->SetPhysData(thePE1Index, result);
};



/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC(GridEdgeData* theData,
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

  Standard_Real DualContourValue2=0.0;
  const vector<T_Element>& theNearMEdges= theData->GetNearMEdges();   //GridEdgeData elements in near slice
  Standard_Size  nb = theNearMEdges.size();
  for(Standard_Integer index=0; index<nb; index ++){
    DualContourValue2 +=
      ( (theNearMEdges[index].GetData())->GetSweptPhysData(theMagPhysIndex) ) *
      ( (theNearMEdges[index].GetData())->GetDualGeomDim_Near() ) *
      (  theNearMEdges[index].GetRelatedDir() );
  }

  Standard_Real C0 = 
    (theData->GetEpsilon() - 0.5*theData->GetSigma()*dt)/
    (theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);
  Standard_Real C2 = dt/(theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);

  Standard_Integer Dir1 = TwoDim_DirBump(theData->GetDir(),1);
  Standard_Real Kappa1 = theData->GetPMLKappa(Dir1);
  Standard_Real result =  
    C0 * theData->GetPhysData(theElecPhysIndex) + 
    C2 * DualContourValue1/Kappa1/theData->GetDualGeomDim() + 
    C2 * DualContourValue2/theData->GetDualGeomDim() + 
    C2 * theData->GetPhysData(thePE1Index);

  theData->SetPhysData(theElecPhysIndex, result);
};



void Advance_CPML_ElecElems_SI_SC(vector<GridEdgeData*>& theDatas,
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
void Advance_CPML_OneElecElem_SI_SC_Damping(GridEdgeData* theData,
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

  Standard_Real preElecPhysData = theData->GetPhysData(theElecPhysIndex);
  Advance_CPML_OneElecElem_SI_SC(theData, Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_Damping_2(theData, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, preElecPhysData);
};


void Advance_CPML_ElecElems_SI_SC_Damping(vector<GridEdgeData*>& theDatas,
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
  Standard_Size zrindex[2];
  Standard_Integer dir;
  //#pragma omp parallel for
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    //theDatas[eindex]->GetBaseGridEdge()->GetVecIndex(zrindex);
    dir = theDatas[eindex]->GetDir();
    if(dir == 0 && zrindex[1] == 1)
    {
    	continue;
    }
    else
    {
    Advance_CPML_OneElecElem_SI_SC_Damping(theDatas[eindex],
					   Dt, si_scale, 
					   theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex, 
					   damping_scale, theAEIndex, theBEIndex);
	}
  }
}





/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC_Damping_new(GridEdgeData* theData,
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



void Advance_CPML_ElecElems_SI_SC_Damping_new(vector<GridEdgeData*>& theDatas,
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
