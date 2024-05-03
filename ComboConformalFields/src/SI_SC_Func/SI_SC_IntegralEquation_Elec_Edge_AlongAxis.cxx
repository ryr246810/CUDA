#include<SI_SC_IntegralEquation.hxx>
#include<GridEdgeData.hxx>
#include <vector>
#include <GridEdge.hxx>


void Compute_DualContourValue_SI_SC_AlongAxis(GridEdgeData* theData, 
				    Standard_Integer theMagPhysIndex,
				    Standard_Real& DualContourValue)
{
  //DualContourValue = 0;

  const vector<T_Element>& theOutLineTElems = theData->GetSharedTFace();  // GridFaceData + dir

  Standard_Integer nb = theOutLineTElems.size(); 

  for(Standard_Integer index=0; index<nb; index ++){
      double tmp = ( (theOutLineTElems[index].GetData())->GetPhysData(theMagPhysIndex) ) *
      ( (theOutLineTElems[index].GetData())->GetDualGeomDim() ) * 
      (  theOutLineTElems[index].GetRelatedDir() ) ;
	   DualContourValue += tmp;
  }  
};



//--------------------------------------------------------->>>
void Advance_OneElecElem_SI_SC_AlongAxis(vector<GridEdgeData*> theDatas,
			       Standard_Size m_PhiNumber,
			       Standard_Real Dt, 
			       Standard_Real si_scale,
			       Standard_Integer theElecPhysIndex,
			       Standard_Integer theJIndex,
			       Standard_Integer theMagPhysIndex)
{

  Standard_Real dt = si_scale*Dt;

  Standard_Real DualContourValue = 0.0;
  for(Standard_Size i=0;i<m_PhiNumber;i++){
  	Compute_DualContourValue_SI_SC_AlongAxis(theDatas[i], 
				       theMagPhysIndex,
				       DualContourValue);
  }

  /*for(Standard_Size i=0;i<m_PhiNumber;i++){
  	Compute_DualContourValue_SI_SC(theDatas[i], 
				       theMagPhysIndex,
				       DualContourValue);
  }*/

  Standard_Real C0 = 0.0 ;
  for(Standard_Size i=0;i<m_PhiNumber;i++){
    C0 += 
    	(theDatas[i]->GetEpsilon() - 0.5*theDatas[i]->GetSigma()*dt)/
    	(theDatas[i]->GetEpsilon() + 0.5*theDatas[i]->GetSigma()*dt);

  }

   C0 /=m_PhiNumber;

  Standard_Real C2 = 0.0 ;
  for(Standard_Size i=0;i<m_PhiNumber;i++){

	C2 += dt/(theDatas[i]->GetEpsilon() + 0.5*theDatas[i]->GetSigma()*dt);
  }

  C2 /=m_PhiNumber;

  Standard_Real PhysData = 0.0 ;
  for(Standard_Size i=0;i<m_PhiNumber;i++){

     PhysData += theDatas[i]->GetPhysData(theElecPhysIndex);
  }

  PhysData /=m_PhiNumber;
   
  Standard_Real JValue= 0;
  for(Standard_Size i=0;i<m_PhiNumber;i++){

	JValue += theDatas[i]->GetPhysData(theJIndex);
  }
  JValue = JValue/m_PhiNumber;

  Standard_Real DualGeomDim= 0;

  for(Standard_Size i=0;i<m_PhiNumber;i++){

	DualGeomDim += theDatas[i]->GetDualGeomDim();
  }

  Standard_Real result = 
    C0 * PhysData+ 
    C2 * (DualContourValue - JValue)/DualGeomDim;
	
  for(Standard_Size i=0;i<m_PhiNumber;i++){
  	theDatas[i]->SetPhysData(theElecPhysIndex, result);
  }	
};



void Advance_ElecElems_SI_SC_AlongAxis(vector<GridEdgeData*>& theDatas,
			     Standard_Size m_PhiNumber,
			     Standard_Real Dt, 
			     Standard_Real si_scale,
			     Standard_Integer theElecPhysIndex, 
			     Standard_Integer theJIndex, 
			     Standard_Integer theMagPhysIndex)
{
  Standard_Size nbe = theDatas.size(); 
  if(nbe % m_PhiNumber !=0)
  {
	std::cout<<"void Advance_ElecElems_SI_SC_AlongAxis --------------------error!!"<<endl;
	//exit(1);
  }
  Standard_Size n_edge = nbe /m_PhiNumber;
  
  for(Standard_Size eindex=0; eindex<n_edge; eindex++){

    vector<GridEdgeData*> theSameEdge;
    theSameEdge.clear();

    for(Standard_Size i=0; i<m_PhiNumber; i++){

    	Standard_Size index = n_edge*i + eindex;

    	theSameEdge.push_back(theDatas[index]);
    }

    Advance_OneElecElem_SI_SC_AlongAxis(theSameEdge,m_PhiNumber,
			      Dt, si_scale,
			      theElecPhysIndex, theJIndex, theMagPhysIndex);
  }
}



void Advance_OneElecElem_SI_SC_AlongAxis_Damping(vector<GridEdgeData*> theDatas,
				       Standard_Size m_PhiNumber,
				       Standard_Real Dt, 
				       Standard_Real si_scale,
				       Standard_Integer theElecPhysIndex,
				       Standard_Integer theJIndex,
				       Standard_Integer theMagPhysIndex,
				       Standard_Real damping_scale,
				       Standard_Integer theAEIndex,
				       Standard_Integer theBEIndex)
{
  Advance_OneElecElem_SI_SC_AlongAxis_Damping_1(theDatas, m_PhiNumber, theElecPhysIndex, damping_scale, theAEIndex);

  vector<Standard_Real> preElecPhysData;
  for(Standard_Size i=0;i<m_PhiNumber;i++)
  {
   preElecPhysData.push_back(theDatas[i]->GetPhysData(theElecPhysIndex));
  }

  Advance_OneElecElem_SI_SC_AlongAxis(theDatas, m_PhiNumber, Dt, si_scale, theElecPhysIndex, theJIndex, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_AlongAxis_Damping_2(theDatas, m_PhiNumber, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, preElecPhysData);
}



void Advance_ElecElems_SI_SC_AlongAxis_Damping(vector<GridEdgeData*>& theDatas,
				     Standard_Size m_PhiNumber,
				     Standard_Real Dt, 
				     Standard_Real si_scale,
				     Standard_Integer theElecPhysIndex,
				     Standard_Integer theJIndex,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real damping_scale,
				     Standard_Integer theAEIndex,
				     Standard_Integer theBEIndex)
{
  Standard_Size nbe = theDatas.size();

  if(nbe % m_PhiNumber !=0)
  {
	std::cout<<"void Advance_ElecElems_SI_SC_AlongAxis_Damping--------------------error!!"<<endl;
	//exit(1);
  }

  Standard_Size n_edge = nbe /m_PhiNumber;
  
  for(Standard_Size eindex=0; eindex< n_edge; eindex++){

    vector<GridEdgeData*> theSameEdge;

    for(Standard_Size i=0; i<m_PhiNumber; i++){

    	Standard_Size index = n_edge*i + eindex;

    	theSameEdge.push_back(theDatas[index]);
    }
    Advance_OneElecElem_SI_SC_AlongAxis_Damping(theSameEdge, m_PhiNumber, Dt, si_scale,
				      theElecPhysIndex, theJIndex, theMagPhysIndex,
				      damping_scale, theAEIndex, theBEIndex);
  }
}





