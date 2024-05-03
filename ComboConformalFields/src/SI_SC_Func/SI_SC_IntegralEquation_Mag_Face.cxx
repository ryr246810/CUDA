#include<SI_SC_IntegralEquation.hxx>
#include<GridEdge.hxx>
// #include "omp.h"


void Compute_ContourValue_SI_SC(GridFaceData* theData, 
				Standard_Integer theElecPhysIndex,
				Standard_Real& ContourValue)
{
  ContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetOutLineTEdge();  //GridEdgeData elements
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
    // #pragma omp atomic
    ContourValue += 
      ( (theOutLineTElems[index].GetData())->GetPhysData(theElecPhysIndex) ) *
      ( (theOutLineTElems[index].GetData())->GetGeomDim() ) *
      (  theOutLineTElems[index].GetRelatedDir() );
  }
};



void Advance_OneMagElem_SI_SC(GridFaceData* theData,
			      Standard_Real Dt, 
			      Standard_Real si_scale,
			      Standard_Integer theMagPhysIndex,
			      Standard_Integer theJMIndex, 
			      Standard_Integer theElecPhysIndex)
{
  Standard_Real C0 = si_scale*Dt/(theData->GetMu());

  Standard_Real ContourValue = 0;

  Compute_ContourValue_SI_SC(theData, theElecPhysIndex,  ContourValue);
  
  Standard_Real result =
    - C0 * ContourValue / theData->GetGeomDim()
    + C0 * theData->GetPhysData(theJMIndex);
  // #pragma omp critical
  {
  theData->AddPhysData(theMagPhysIndex, result);
  }
};


void Advance_MagElems_SI_SC(vector<GridFaceData*>& theDatas, 
			    Standard_Real Dt, 
			    Standard_Real si_scale,
			    Standard_Integer theMagPhysIndex, 
			    Standard_Integer theJMIndex, 
			    Standard_Integer theElecPhysIndex)
{
  Standard_Size nbf = theDatas.size();
  // printf("\n theDatas.size() is %d \n", nbf);
  // #pragma omp parallel for
  for(Standard_Size findex=0; findex<nbf; findex++){
    Advance_OneMagElem_SI_SC(theDatas[findex],  
			     Dt, si_scale, 
			     theMagPhysIndex, theJMIndex, 
			     theElecPhysIndex);
  }
}

