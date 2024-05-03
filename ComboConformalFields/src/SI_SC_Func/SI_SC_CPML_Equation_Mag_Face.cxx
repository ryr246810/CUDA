#include <SI_SC_CPML_Equation.hxx>
// #include "omp.h"


/******************************************************************************************************/
void Compute_ContourValue1_SI_SC(GridFaceData* theData, 
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue)
{
  contourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetOutLineTEdge();
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
    GridEdgeData* currEdgeData = (GridEdgeData*) theOutLineTElems[index].GetData();
    Standard_Integer currRelativeDir = theOutLineTElems[index].GetRelatedDir();

    contourValue += 
      currEdgeData->GetPhysData(theElecPhysIndex) *
      currEdgeData->GetGeomDim() *
      currRelativeDir *
      (currEdgeData->GetDir() - ThreeDim_DirBump(2,1)) / (ThreeDim_DirBump(2,2) - ThreeDim_DirBump(2,1));
  }
};



/******************************************************************************************************/
void Compute_ContourValue2_SI_SC(GridFaceData* theData,
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue)
{
  contourValue = 0;
  const vector<T_Element>& theOutLineTElems = theData->GetOutLineTEdge();
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
    GridEdgeData* currEdgeData = (GridEdgeData*) theOutLineTElems[index].GetData();
    Standard_Integer currRelativeDir = theOutLineTElems[index].GetRelatedDir();

    contourValue += 
      currEdgeData->GetPhysData(theElecPhysIndex) *
      currEdgeData->GetGeomDim() *
      currRelativeDir *
      (currEdgeData->GetDir() - ThreeDim_DirBump(2,2)) / (ThreeDim_DirBump(2,1) - ThreeDim_DirBump(2,2));
  }
};



/******************************************************************************************************/
void Compute_PM1_SI_SC(GridFaceData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePM1Index, 
		       Standard_Real contourValue)
{
  Standard_Integer theTruncDir = ThreeDim_DirBump(2,1);

  Standard_Real a=0;
  Standard_Real b=0;

  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * (theData->GetPhysData(thePM1Index)) + 
    a * contourValue / theData->GetGeomDim();

  theData->SetPhysData(thePM1Index, result);
};



/******************************************************************************************************/
void Compute_PM2_SI_SC(GridFaceData* theData, 
		       Standard_Real Dt, 
		       Standard_Integer thePM2Index,
		       Standard_Real contourValue)
{
  Standard_Integer theTruncDir = ThreeDim_DirBump(2,2);

  Standard_Real a=0;
  Standard_Real b=0;

  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * (theData->GetPhysData(thePM2Index)) + 
    a * contourValue / theData->GetGeomDim();

  theData->SetPhysData(thePM2Index, result);
};



/******************************************************************************************************/
void Advance_CPML_OneMagElem_SI_SC(GridFaceData* theData, 
				   Standard_Real Dt, 
				   Standard_Real si_scale,
				   Standard_Integer theMagPhysIndex,
				   Standard_Integer thePM1Index, 
				   Standard_Integer thePM2Index, 
				   Standard_Integer theElecPhysIndex)
{
  Standard_Real dt = si_scale*Dt;
  Standard_Real C0 =  dt/(theData->GetMu());

  Standard_Real contourValue1;
  Compute_ContourValue1_SI_SC(theData, theElecPhysIndex, contourValue1);

  Standard_Real contourValue2;
  Compute_ContourValue2_SI_SC(theData, theElecPhysIndex, contourValue2);
  
  Standard_Integer Dir1 = ThreeDim_DirBump(2,1);
  Standard_Real Kappa1 = theData->GetPMLKappa(Dir1);

  Standard_Integer Dir2 = ThreeDim_DirBump(2,2);
  Standard_Real Kappa2 = theData->GetPMLKappa(Dir2);

  Compute_PM1_SI_SC(theData, dt, thePM1Index, contourValue1);
  Compute_PM2_SI_SC(theData, dt, thePM2Index, contourValue2);

  Standard_Real result = 
    theData->GetPhysData(theMagPhysIndex)
    - C0 * (contourValue1/Kappa1 + contourValue2/Kappa2)/(theData->GetGeomDim())
    - C0 * theData->GetPhysData(thePM1Index) 
    - C0 * theData->GetPhysData(thePM2Index);

  theData->SetPhysData(theMagPhysIndex, result);
};


/******************************************************************************************************/
void Advance_CPML_MagElems_SI_SC(vector<GridFaceData*>& theDatas, 
				 Standard_Real Dt, 
				 Standard_Real si_scale,
				 Standard_Integer theMagPhysIndex, 
				 Standard_Integer thePM1Index, 
				 Standard_Integer thePM2Index, 
				 Standard_Integer theElecPhysIndex)
{
  Standard_Size nbf = theDatas.size();
  //#pragma omp parallel for		
  for(Standard_Size findex=0; findex<nbf; findex++){
    Advance_CPML_OneMagElem_SI_SC(theDatas[findex], Dt, si_scale, 
				  theMagPhysIndex, thePM1Index, 
				  thePM2Index, theElecPhysIndex);
  } 
}
