#include <SI_SC_CPML_Equation.hxx>
#include <SI_SC_IntegralEquation.hxx>


/****************************************************************************************/
void Compute_DualContourValue1_SI_SC_AlongAxis(GridEdgeData* theData,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue)
{
  //DualContourValue = 0;
  Standard_Integer nb = theData->GetSharedTFace().size();
  for(Standard_Integer index=0; index<nb; index ++){
    DualContourValue += 
      ( ((theData->GetSharedTFace()[index]).GetData())->GetPhysData(theMagPhysIndex) ) *
      ( ((theData->GetSharedTFace()[index]).GetData())->GetDualGeomDim() ) *
      (  (theData->GetSharedTFace()[index]).GetRelatedDir() ) ;
  }

};



/****************************************************************************************/
void Compute_PE1_SI_SC_AlongAxis(vector<GridEdgeData*> theDatas,
	               Standard_Size m_PhiNumber,
		       Standard_Real Dt,
		       Standard_Integer thePE1Index,
		       Standard_Real DualContourValue)
{
  Standard_Real result = 0.0;
  Standard_Real PhysData = 0.0;
  Standard_Real DualGeomDim= 0.0;

  for(Standard_Size i=0;i<m_PhiNumber;i++){

	DualGeomDim += theDatas[i]->GetDualGeomDim();
  }
  Standard_Real a=0.0,b=0.0;
  for(Standard_Size i=0;i<m_PhiNumber; i++){
  	Standard_Integer theTruncDir = TwoDim_DirBump(theDatas[i]->GetDir(),1);

  	Standard_Real a_tmp = 0.0;
  	Standard_Real b_tmp = 0.0;
  	Get_a_b_SI_SC(theDatas[i], theTruncDir, Dt, a_tmp, b_tmp);
  	PhysData +=theDatas[i]->GetPhysData(thePE1Index);
  	DualGeomDim += theDatas[i]->GetDualGeomDim();
	a += a_tmp;
	b += b_tmp;
  }

  PhysData /=m_PhiNumber;
  a /=m_PhiNumber;
  b /=m_PhiNumber;
  

  result = b * PhysData + a* DualContourValue / DualGeomDim;
  

  for(Standard_Size i=0;i<m_PhiNumber; i++){

  	theDatas[i]->SetPhysData(thePE1Index, result);
  }
};



/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC_AlongAxis(vector<GridEdgeData*> theDatas,
	                            Standard_Size m_PhiNumber,
				    Standard_Real Dt, 
				    Standard_Real si_scale,
				    Standard_Integer theElecPhysIndex,
				    Standard_Integer thePE1Index,
				    Standard_Integer thePE2Index,
				    Standard_Integer theMagPhysIndex)
{
  Standard_Real dt = si_scale*Dt;

  Standard_Real DualContourValue1 = 0.0;
  for(Standard_Size i=0;i<m_PhiNumber;i++){
  	Compute_DualContourValue1_SI_SC_AlongAxis(theDatas[i], theMagPhysIndex, DualContourValue1);
  }

  Compute_PE1_SI_SC_AlongAxis(theDatas, m_PhiNumber, dt, thePE1Index, DualContourValue1);
  

  Standard_Real C0 = 0.0 ; 
  Standard_Real C2 = 0.0 ;
  for(Standard_Size i=0;i<m_PhiNumber;i++){
    C0 += (theDatas[i]->GetEpsilon() - 0.5*theDatas[i]->GetSigma()*dt)/
    (theDatas[i]->GetEpsilon() + 0.5*theDatas[i]->GetSigma()*dt);

    C2 += dt/(theDatas[i]->GetEpsilon() + 0.5*theDatas[i]->GetSigma()*dt);
  }

  C0 /= m_PhiNumber ;
  C2 /=m_PhiNumber;

  Standard_Real Kappa1   = 0.0;
  Standard_Real PhysData = 0.0;
  Standard_Real PhysData_PE = 0.0;
  Standard_Real DualGeomDim = 0.0;

  for(Standard_Size i=0;i<m_PhiNumber;i++){
  	Standard_Integer Dir1 = TwoDim_DirBump(theDatas[i]->GetDir(),1);
 	Kappa1 += theDatas[i]->GetPMLKappa(Dir1);
	PhysData += theDatas[i]->GetPhysData(theElecPhysIndex);
	PhysData_PE += theDatas[i]->GetPhysData(thePE1Index);
	DualGeomDim += theDatas[i]->GetDualGeomDim();
   }

   Kappa1 /=m_PhiNumber;
   PhysData /=m_PhiNumber;
   PhysData_PE /=m_PhiNumber;

  Standard_Real result =  
    C0 * PhysData+ C2 * DualContourValue1/Kappa1/DualGeomDim+C2 * PhysData_PE;

  for(Standard_Size i=0;i<m_PhiNumber;i++){
  	theDatas[i]->SetPhysData(theElecPhysIndex, result);
   }

};



void Advance_CPML_ElecElems_SI_SC_AlongAxis(vector<GridEdgeData*>& theDatas,
	                          Standard_Size m_PhiNumber,
				  Standard_Real Dt, 
				  Standard_Real si_scale,
				  Standard_Integer theElecPhysIndex,
				  Standard_Integer thePE1Index,
				  Standard_Integer thePE2Index,
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

    for(Standard_Size i=0; i<m_PhiNumber; i++){

    	Standard_Size index = n_edge*i + eindex;

    	theSameEdge.push_back(theDatas[index]);
    }
    Advance_CPML_OneElecElem_SI_SC_AlongAxis(theSameEdge, m_PhiNumber, Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);
  }
}



/****************************************************************************************/
void Advance_CPML_OneElecElem_SI_SC_AlongAxis_Damping(vector<GridEdgeData*> theDatas,
	                                    Standard_Size m_PhiNumber,
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
  Advance_OneElecElem_SI_SC_AlongAxis_Damping_1(theDatas,m_PhiNumber, theElecPhysIndex, damping_scale, theAEIndex);

  vector<Standard_Real> preElecPhysData;
  for(Standard_Size i=0;i<m_PhiNumber;i++)
  {
   preElecPhysData.push_back(theDatas[i]->GetPhysData(theElecPhysIndex));
  }
  Advance_CPML_OneElecElem_SI_SC_AlongAxis(theDatas, m_PhiNumber, Dt, si_scale, theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex);

  Advance_OneElecElem_SI_SC_AlongAxis_Damping_2(theDatas,m_PhiNumber, theElecPhysIndex, damping_scale, theAEIndex, theBEIndex, preElecPhysData);
};



void Advance_CPML_ElecElems_SI_SC_AlongAxis_Damping(vector<GridEdgeData*>& theDatas,
	                                  Standard_Size m_PhiNumber,
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

  if(nbe % m_PhiNumber !=0)
  {
	std::cout<<"void Advance_ElecElems_SI_SC_AlongAxis_Damping--------------------error!!"<<endl;
	//exit(1);
  }

  Standard_Size n_edge = nbe /m_PhiNumber;
  
  for(Standard_Size eindex=0; eindex<n_edge; eindex++){

    vector<GridEdgeData*> theSameEdge;

    for(Standard_Size i=0; i<m_PhiNumber; i++){

    	Standard_Size index = n_edge*i + eindex;

    	theSameEdge.push_back(theDatas[index]);
    }
    Advance_CPML_OneElecElem_SI_SC_AlongAxis_Damping(theSameEdge, m_PhiNumber,
					   Dt, si_scale, 
					   theElecPhysIndex, thePE1Index, thePE2Index, theMagPhysIndex, 
					   damping_scale, theAEIndex, theBEIndex);
  }
}
