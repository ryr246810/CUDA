#include<SI_SC_IntegralEquation.hxx>
#include<GridVertexData.hxx>
#include<GridEdgeData.hxx>
#include<GridFaceData.cuh>
// #include "omp.h"

void Compute_DualContourValue_SI_SC(GridVertexData* theData, 
				    Standard_Integer theMagPhysIndex,
				    Standard_Real& DualContourValue)
{
  DualContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetSharedTDFaces(); // GridEdgeData + dir

  Standard_Integer nb = theOutLineTElems.size();
  for(Standard_Integer index=0; index<nb; index ++){
    // #pragma omp atomic
    DualContourValue += 
      (theOutLineTElems[index].GetData())->GetSweptPhysData(theMagPhysIndex) *
      (theOutLineTElems[index].GetData())->GetDualSweptGeomDim() * 
      theOutLineTElems[index].GetRelatedDir();
  }
};


//--------------------------------------------------------->>>
void Advance_OneElecElem_SI_SC(GridVertexData* theData, 
			       Standard_Real Dt, 
			       Standard_Real si_scale,
			       Standard_Integer theElecPhysIndex,
			       Standard_Integer theCurrentIndex,
			       Standard_Integer theMagPhysIndex)
{
  Standard_Real dt = si_scale*Dt;

  Standard_Real DualContourValue = 0;
  Compute_DualContourValue_SI_SC(theData, 
				 theMagPhysIndex,
				 DualContourValue);

  Standard_Real C0 = 
    (theData->GetEpsilon() - 0.5*theData->GetSigma()*dt)/
    (theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);

  Standard_Real C2 = dt/(theData->GetEpsilon() + 0.5*theData->GetSigma()*dt);
  Standard_Real result = 
    C0 * theData->GetSweptPhysData(theElecPhysIndex)
    + C2 * (DualContourValue - theData->GetSweptPhysData(theCurrentIndex))/(theData->GetDualSweptGeomDim());
    //+ C2 * DualContourValue/(theData->GetDualSweptGeomDim())
    //- C2 * theData->GetSweptPhysData(theCurrentIndex);
  theData->SetSweptPhysData(theElecPhysIndex, result);
};


void Advance_ElecElems_SI_SC(vector<GridVertexData*>& theDatas,
			     Standard_Real Dt, 
			     Standard_Real si_scale,
			     Standard_Integer theElecPhysIndex, 
			     Standard_Integer theCurrentIndex, 
			     Standard_Integer theMagPhysIndex)
{
  Standard_Size nbe = theDatas.size();
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_OneElecElem_SI_SC(theDatas[eindex],  
			      Dt, 
			      si_scale,
			      theElecPhysIndex, 
			      theCurrentIndex, 
			      theMagPhysIndex);
  }
}



void Advance_OneElecElem_SI_SC_Damping(GridVertexData* theData, 
				       Standard_Real Dt, 
				       Standard_Real si_scale,
				       Standard_Integer theElecPhysIndex,
				       Standard_Integer theCurrentIndex,
				       Standard_Integer theMagPhysIndex,
				       Standard_Real damping_scale,
				       Standard_Integer theAEIndex,
				       Standard_Integer theBEIndex)
{
  Advance_OneElecElem_SI_SC_Damping_1(theData, theElecPhysIndex, damping_scale, theAEIndex);

  Standard_Real preElecPhysData = theData->GetSweptPhysData(theElecPhysIndex);
  Advance_OneElecElem_SI_SC(theData, Dt, si_scale, theElecPhysIndex, theCurrentIndex, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_Damping_2(theData, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, preElecPhysData);
}


void Advance_ElecElems_SI_SC_Damping(vector<GridVertexData*>& theDatas,
				     Standard_Real Dt, 
				     Standard_Real si_scale,
				     Standard_Integer theElecPhysIndex,
				     Standard_Integer theCurrentIndex,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real damping_scale,
				     Standard_Integer theAEIndex,
				     Standard_Integer theBEIndex)
{
  Standard_Size nbe = theDatas.size();
//   #pragma omp parallel for
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_OneElecElem_SI_SC_Damping(theDatas[eindex],
				      Dt, si_scale,
				      theElecPhysIndex, theCurrentIndex, theMagPhysIndex,
				      damping_scale, theAEIndex, theBEIndex);
  
  }
}



void Advance_OneElecElem_SI_SC_Damping_new(GridVertexData* theData, 
					   Standard_Real Dt, 
					   Standard_Real si_scale,
					   Standard_Integer theElecPhysIndex,
					   Standard_Integer theCurrentIndex,
					   Standard_Integer theMagPhysIndex,
					   Standard_Real damping_scale,
					   Standard_Integer theAEIndex,
					   Standard_Integer theBEIndex,
					   Standard_Integer thePREIndex)
{
  Advance_OneElecElem_SI_SC_Damping_3(theData, theElecPhysIndex, damping_scale, theAEIndex, thePREIndex);
  Advance_OneElecElem_SI_SC(theData, Dt, si_scale, theElecPhysIndex, theCurrentIndex, theMagPhysIndex);
  Advance_OneElecElem_SI_SC_Damping_4(theData, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, thePREIndex);
}


void Advance_ElecElems_SI_SC_Damping_new(vector<GridVertexData*>& theDatas,
					 Standard_Real Dt, 
					 Standard_Real si_scale,
					 Standard_Integer theElecPhysIndex,
					 Standard_Integer theCurrentIndex,
					 Standard_Integer theMagPhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Integer thePREIndex)
{
  Standard_Size nbe = theDatas.size();
  for(Standard_Size eindex=0; eindex<nbe; eindex++){
    Advance_OneElecElem_SI_SC_Damping_new(theDatas[eindex],
					  Dt, 
					  si_scale,
					  theElecPhysIndex,
					  theCurrentIndex,
					  theMagPhysIndex,
					  damping_scale,
					  theAEIndex,
					  theBEIndex,
					  thePREIndex);
  }
}
