#include <SI_SC_CPML_Equation.hxx>
// #include "omp.h"

/******************************************************************************************************/
void Compute_ContourValue1_SI_SC(GridEdgeData* theData, 
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue)
{
  contourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetOutLineDTEdges();  //GridVertexData elements
  Standard_Integer nb = theOutLineTElems.size();
  for(Standard_Integer index=0; index<nb; index ++){
    contourValue += 
      ( (theOutLineTElems[index].GetData())->GetSweptPhysData(theElecPhysIndex) ) *
      ( (theOutLineTElems[index].GetData())->GetSweptGeomDim() ) *
      (  theOutLineTElems[index].GetRelatedDir() );
  }


};


/******************************************************************************************************/
void Compute_PM1_SI_SC(GridEdgeData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePM1Index, 
		       Standard_Real contourValue)
{
  Standard_Integer theTruncDir = TwoDim_DirBump(TwoDim_DirBump(theData->GetDir(),1), 1);

  Standard_Real a=0;
  Standard_Real b=0;

  Get_a_b_SI_SC(theData, theTruncDir, Dt, a, b);

  Standard_Real result = 
    b * (theData->GetSweptPhysData(thePM1Index)) + 
    a * contourValue / theData->GetSweptGeomDim();

  theData->SetSweptPhysData(thePM1Index, result);
};



/******************************************************************************************************/
void Advance_CPML_OneMagElem_SI_SC(GridEdgeData* theData, 
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

  Standard_Real contourValue2=0.0;
  const vector<T_Element>& theNearEEdges= theData->GetNearEEdges();   //GridEdgeData elements in near slice
  Standard_Size nb = theNearEEdges.size();
  for(Standard_Integer index=0; index<nb; index ++){
    contourValue2 +=
      ( (theNearEEdges[index].GetData())->GetPhysData(theElecPhysIndex) ) *
      ( (theNearEEdges[index].GetData())->GetSweptGeomDim_Near() ) *
      (  theNearEEdges[index].GetRelatedDir() );
  }
  
  Standard_Integer Dir1 = TwoDim_DirBump(TwoDim_DirBump(theData->GetDir(),1), 1);
  Standard_Real Kappa1 = theData->GetPMLKappa(Dir1);

  Compute_PM1_SI_SC(theData, dt, thePM1Index, contourValue1);
  
  Standard_Real result = 
    theData->GetSweptPhysData(theMagPhysIndex)
    - C0 * contourValue1/Kappa1/theData->GetSweptGeomDim()
    - C0 * contourValue2/theData->GetSweptGeomDim()
    //- C0 * contourValue1/Kappa1/theData->GetBaseGridSweptGeomDim() //tzh Modify 20210412
    //- C0 * contourValue2/theData->GetBaseGridSweptGeomDim()
    - C0 * theData->GetSweptPhysData(thePM1Index);

  theData->SetSweptPhysData(theMagPhysIndex, result);
};


/******************************************************************************************************/
void Advance_CPML_MagElems_SI_SC(vector<GridEdgeData*>& theDatas, 
				 Standard_Real Dt, 
				 Standard_Real si_scale,
				 Standard_Integer theMagPhysIndex, 
				 Standard_Integer thePM1Index, 
				 Standard_Integer thePM2Index, 
				 Standard_Integer theElecPhysIndex)
{
  Standard_Size nb = theDatas.size();
  //#pragma omp parallel for
  for(Standard_Size findex=0; findex<nb; findex++){
    Advance_CPML_OneMagElem_SI_SC(theDatas[findex], 
				  Dt, si_scale, 
				  theMagPhysIndex, thePM1Index, thePM2Index, theElecPhysIndex);
  }
}
