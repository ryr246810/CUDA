#include<SI_SC_IntegralEquation.hxx>
#include<GridEdge.hxx>
// #include "omp.h"


void Compute_ContourValue_SI_SC(GridEdgeData* theData, 
				Standard_Integer theElecPhysIndex,
				Standard_Real& ContourValue)
{
  ContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetOutLineDTEdges();   //GridVertexData elements
  Standard_Integer nb = theOutLineTElems.size();

  for(Standard_Integer index=0; index<nb; index ++){
  	// #pragma omp atomic
    ContourValue += 
      ( (theOutLineTElems[index].GetData())->GetSweptPhysData(theElecPhysIndex) ) *
      ( (theOutLineTElems[index].GetData())->GetSweptGeomDim() ) *
      (  theOutLineTElems[index].GetRelatedDir() );
  }
  const vector<T_Element>& theNearEEdges= theData->GetNearEEdges();   //GridEdgeData elements in near slice
  nb = theNearEEdges.size();
  for(Standard_Integer index=0; index<nb; index ++){
  	// #pragma omp atomic
    ContourValue +=
      ( (theNearEEdges[index].GetData())->GetPhysData(theElecPhysIndex) ) *
      ( (theNearEEdges[index].GetData())->GetSweptGeomDim_Near() ) *
      (  theNearEEdges[index].GetRelatedDir() );
  }

};



void Advance_OneMagElem_SI_SC(GridEdgeData* theData,
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
    - C0 * ContourValue / theData->GetSweptGeomDim()
    //- C0 * ContourValue / theData->GetBaseGridSweptGeomDim()  //tzh Modify 20210412
    + C0 * theData->GetSweptPhysData(theJMIndex);
  // #pragma omp critical
  {
  theData->AddSweptPhysData(theMagPhysIndex, result);
  }
};



void Advance_MagElems_SI_SC(vector<GridEdgeData*>& theDatas, 
			    Standard_Real Dt, 
			    Standard_Real si_scale,
			    Standard_Integer theMagPhysIndex, 
			    Standard_Integer theJMIndex, 
			    Standard_Integer theElecPhysIndex)
{
  Standard_Size nbf = theDatas.size();
  Standard_Size zrindex[2];
  // #pragma omp parallel for private(zrindex)
  for(Standard_Size findex=0; findex<nbf; findex++){
    theDatas[findex]->GetBaseGridEdge()->GetVecIndex(zrindex);
    //if(zrindex[1]==1) cout<<zrindex[1]<<endl;
    Advance_OneMagElem_SI_SC(theDatas[findex],  
			     Dt, si_scale, 
			     theMagPhysIndex, theJMIndex, 
			     theElecPhysIndex);
  
  }
}

