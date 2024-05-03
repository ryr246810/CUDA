#include <ZRGrid.hxx>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <PhysConsts.hxx>

void 
ZRGrid::
ComputeIndexVecAndWeightsInGrid(const TxVector2D<Standard_Real>& pos, 
				TxVector2D<Standard_Size>& indx, 
				TxVector2D<Standard_Real>& wl,
				TxVector2D<Standard_Real>& wu) const
{
  Standard_Size tmpIndx[2];
  Standard_Real tmpWl[2];
  Standard_Real tmpWu[2];

  ComputeIndexVecAndWeightsInGrid(pos,tmpIndx,tmpWl,tmpWu);

  for(Standard_Size i=0; i<2; i++){
    indx[i] = tmpIndx[i];
    wl[i] = tmpWl[i];
    wu[i] = tmpWu[i];
  }
}


void 
ZRGrid::
ComputeIndexVecAndWeightsInGrid(Standard_Real pos[2], 
				Standard_Size indx[2], 
				Standard_Real wl[2],
				Standard_Real wu[2]) const
{
  TxVector2D<Standard_Real> tmpPos(pos);
  ComputeIndexVecAndWeightsInGrid(tmpPos, indx, wl, wu);
}


void 
ZRGrid::
ComputeIndexVecAndWeightsInGrid(const TxVector2D<Standard_Real>& pos, 
				Standard_Size indx[2], 
				Standard_Real wl[2],
				Standard_Real wu[2]) const
{
  Standard_Real dl[2];
  this->ComputeLocationInGrid(pos, indx, dl);

  TxVector2D<Standard_Real> thisSteps = this->GetSteps(indx);

  for(Standard_Size i=0; i<2; i++){
    wu[i] = dl[i] / thisSteps[i];
    wl[i] = 1.0 - wu[i]; 
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      Standard_Size theIndxVec[2]) const
{
  Standard_Size theIndx;
  TxVector2D<Standard_Real> lengthVec = aLocation - m_Org;
  for(Standard_Integer aDir=0;aDir<2;aDir++){
    ComputeIndex(aDir, lengthVec[aDir],theIndx);
    theIndxVec[aDir] = theIndx;
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      TxVector2D<Standard_Size>& theIndxVec) const
{
  Standard_Size tmpIndxVec[2];
  ComputeLocationInGrid(aLocation, tmpIndxVec);
  for(Standard_Integer aDir=0;aDir<2;aDir++){
    theIndxVec[aDir] = tmpIndxVec[aDir];
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const Standard_Real aLocation[2], 
		      Standard_Size theIndxVec[2]) const
{
  TxVector2D<Standard_Real> tmpLocation(aLocation);
  ComputeLocationInGrid(tmpLocation, theIndxVec);
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      TxVector2D<Standard_Size>& theIndxVec, 
		      TxVector2D<Standard_Real>& thedLVec) const
{
  Standard_Size tmpIndxVec[2];
  Standard_Real tmpdLVec[2];
  ComputeLocationInGrid(aLocation, tmpIndxVec, tmpdLVec);
  for(Standard_Integer aDir=0; aDir<2; aDir++){
    theIndxVec[aDir] = tmpIndxVec[aDir];
    thedLVec[aDir] = tmpdLVec[aDir];
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const Standard_Real aLocation[2], 
		      Standard_Size theIndxVec[2],
		      Standard_Real thedLVec[2]) const
{
  TxVector2D<Standard_Real> tmpLocation(aLocation);
  ComputeLocationInGrid(tmpLocation, theIndxVec, thedLVec); // modified 2016.10.24
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      Standard_Size theIndxVec[2],
		      Standard_Real thedLVec[2]) const
{
  Standard_Size theIndx;
  Standard_Real thedL;
  TxVector2D<Standard_Real> lengthVec = aLocation - m_Org;
  for(Standard_Integer aDir=0; aDir<2; aDir++){
    Standard_Real aL = lengthVec[aDir];
    ComputeIndex(aDir,aL,theIndx);
    theIndxVec[aDir] = theIndx;
    thedLVec[aDir] = aL - GetLength(aDir,theIndx);
  }
}


void 
ZRGrid::
ComputeLocationInGridInDir(const Standard_Integer& aDir, 
			   const Standard_Real& aLocation, 
			   Standard_Size& theIndx) const
{
  Standard_Real theLength = aLocation - m_Org[aDir];

  ComputeIndex(aDir, theLength, theIndx);
}




void 
ZRGrid::
ComputeLocationInGridInDir(const Standard_Integer& aDir, 
			   const Standard_Real& aLocation, 
			   bool& beInRgn,
			   Standard_Size& theIndx,
			   Standard_Size& theFrac) const
{
  beInRgn = false;
  Standard_Real theLength = aLocation - m_Org[aDir];

  if( IsIn(aDir, theLength) ) {
    beInRgn = true;
    ComputeIndex2(aDir, theLength, theIndx, theFrac);
  }else{
    cout<<"ZRGrid::ComputeLocationInGridInDir--------------------------error"<<endl;
  }
}



/****************************************************************/
/* Function : ComputeIndex 
 * Property : Private Function
 * Purpose  : use to compute "theIndex"
 */
/****************************************************************/


#include <vector>

void 
ZRGrid::
ComputeIndex(const Standard_Integer aDir, 
	     const Standard_Real aL, 
	     Standard_Size& theIndex ) const
{
  const vector<Standard_Real>& currVec = m_LVectors.at(aDir);
  vector<Standard_Real>::const_iterator iter = std::upper_bound(currVec.begin(), currVec.end(), aL);
  if( iter!=currVec.end() ){
    Standard_Size tmpIndex = iter - currVec.begin();
    if(tmpIndex!=0){
      theIndex = tmpIndex-1;
    }else{
      theIndex = tmpIndex;
    }
  }else{
    cout<<"error-----------------ZRGrid::ComputeIndex-----------out of region---------------------!!!"<<endl;
    if(aL<0.0){
      theIndex = 0;
    }else if(aL>GetLength(aDir)){
      theIndex = GetDimension(aDir);
    }
  }
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeIndex2(const Standard_Integer aDir,
	      const Standard_Real aL,
	      Standard_Size& theIndex,
	      Standard_Real& thedL) const
{
  ComputeIndex(aDir,aL,theIndex);
  thedL = aL - GetLength(aDir,theIndex);
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeIndex2(const Standard_Integer aDir,
	      const Standard_Real aL, 
	      Standard_Size& theIndex,
	      Standard_Size& theFrac) const
{
  Standard_Real thedL;

  ComputeIndex2(aDir,aL,theIndex,thedL);

  Standard_Real currStep = GetStep(aDir,theIndex);
  Standard_Real theResolution = currStep/m_Resolution;
  Standard_Real theHalfResolution = 0.5 * theResolution;

  if(thedL<m_Tol){
    theFrac = 0;
  }else if( (currStep - thedL) < m_Tol){
    theFrac = GetResolutionRatio();
  }else{
    theFrac = floor(thedL/theResolution);
    Standard_Real tmp = fabs(thedL-theFrac * theResolution);
    if(tmp > theHalfResolution){
      theFrac+=1;
    }
    if( theFrac == 0 ){
      theFrac+=1;
    }else if( theFrac == GetResolutionRatio() ){
      theFrac-=1;
    }else{
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
ZRGrid::
ComputeLocationOfEdgeBndPnt(const Standard_Integer aDir, 
			    const TxVector2D<Standard_Real>& thePnt,
			    bool& beInRgn,
			    Standard_Size& theGridEdgeIndex,
			    Standard_Size& theFrac) const
{
  beInRgn = false;
  theGridEdgeIndex = 0;
  theFrac = 0;

  Standard_Real aLength = 0;
  aLength = thePnt[aDir] - m_Org[aDir];

  if( IsIn(Standard_Integer(aDir), aLength) ) {
    beInRgn = true;
    ComputeIndex2(Standard_Integer(aDir), aLength, theGridEdgeIndex, theFrac);
  }else{
    cout<<"ZRGrid::ComputeGridLocationOfEdgeBndPnt----------------------out of region"<<endl;
  }
}


void 
ZRGrid::
ComputeLocationOfEdgeBndPnt(const Standard_Integer aDir, 
			    const Standard_Real theZRPnt[2], 
			    bool& beInRgn, 
			    Standard_Size& theGridEdgeIndex, 
			    Standard_Size& theFrac) const
{
  TxVector2D<Standard_Real> thePnt(theZRPnt[0], theZRPnt[1]);
  ComputeLocationOfEdgeBndPnt(aDir, thePnt, beInRgn, theGridEdgeIndex, theFrac);
}

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/

void 
ZRGrid::
ComputeLocationOfFaceBndPnt(const TxVector2D<Standard_Real>& thePnt,
			    bool& beInRgn,
			    Standard_Size& theIndex1, 
			    Standard_Size& theFrac1, 
			    Standard_Size& theIndex2,
			    Standard_Size& theFrac2) const
{
  theFrac1 = 0;
  theFrac2 = 0;
  beInRgn = false;

  Standard_Integer Dir1 = 0;
  Standard_Integer Dir2 = 1;

  Standard_Real aLength1 = thePnt[Dir1] - m_Org[Dir1];
  Standard_Real aLength2 = thePnt[Dir2] - m_Org[Dir2];
  
  if( IsIn(Dir1, aLength1) && IsIn(Dir2, aLength2) ) {
    beInRgn = true;
    ComputeIndex2(Dir1,aLength1,theIndex1,theFrac1);
    ComputeIndex2(Dir2,aLength2,theIndex2,theFrac2);
  }else{
    cout<<"ZRGrid::ComputeGridLocationOfFaceBndPnt-----------------out of rgn"<<endl;
  }
}

  
void 
ZRGrid::
ComputeLocationOfFaceBndPnt(const Standard_Real theZRPnt[2],
			    bool& beInRgn,
			    Standard_Size& theIndex1,
			    Standard_Size& theFrac1,
			    Standard_Size& theIndex2,
			    Standard_Size& theFrac2) const
{
  TxVector2D<Standard_Real> thePnt(theZRPnt[0], theZRPnt[1]);
  ComputeLocationOfFaceBndPnt(thePnt, beInRgn, theIndex1, theFrac1, theIndex2, theFrac2);
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      TxVector2D<Standard_Size>& theIndxVec, 
		      TxVector2D<Standard_Size>& theFracVec) const
{

  Standard_Size theIndx;
  Standard_Size theFrac;
  TxVector2D<Standard_Real> lengthVec = aLocation - m_Org;
  for(Standard_Integer aDir=0; aDir<2; aDir++){
    ComputeIndex2(aDir, lengthVec[aDir],theIndx,theFrac);
    theIndxVec[aDir] = theIndx;
    theFracVec[aDir] = theFrac;
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const Standard_Real aLocation[2], 
		      Standard_Size theIndxVec[2],
		      Standard_Size theFracVec[2]) const
{
  TxVector2D<Standard_Real> tmpLocation(aLocation[0], aLocation[1]); 
  TxVector2D<Standard_Size> tmpIndxVec; 
  TxVector2D<Standard_Size> tmpFracVec; 

  ComputeLocationInGrid(tmpLocation, tmpIndxVec, tmpFracVec);
  for(Standard_Integer aDir=0;aDir<2;aDir++){
    theIndxVec[aDir] = tmpIndxVec[aDir];
    theFracVec[aDir] = tmpFracVec[aDir];
  }
}


void 
ZRGrid::
ComputeLocationInGrid(const TxVector2D<Standard_Real>& aLocation, 
		      Standard_Size theIndxVec[2],
		      Standard_Size theFracVec[2]) const
{
  TxVector2D<Standard_Size> tmpIndxVec; 
  TxVector2D<Standard_Size> tmpFracVec; 

  ComputeLocationInGrid(aLocation, tmpIndxVec, tmpFracVec);
  for(Standard_Integer aDir=0;aDir<2;aDir++){
    theIndxVec[aDir] = tmpIndxVec[aDir];
    theFracVec[aDir] = tmpFracVec[aDir];
  }
}
//////////////////These Functions are for the NodeField_Cyl3D  for computating in 3D  ////////////////

void 
ZRGrid::
ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real>& pos, 
				       TxVector<Standard_Size>& indx, 
				       TxVector<Standard_Real>& wl,
				       TxVector<Standard_Real>& wu) const
{

  Standard_Size tmpIndx[2];
  Standard_Real tmpWl[2];
  Standard_Real tmpWu[2];
  Standard_Real phiFac;

  TxVector2D<Standard_Real> tmpPos(pos[0],pos[1]);

  ComputeIndexVecAndWeightsInGrid(tmpPos,tmpIndx,tmpWl,tmpWu);

  for(Standard_Size i=0; i<2; i++){
    indx[i] = tmpIndx[i];
    wl[i] = tmpWl[i];
    wu[i] = tmpWu[i];
  }

  ComputeLocationInGridPhi(pos,indx[2],phiFac);
  wl[2] = 1.0-phiFac;
  wu[2] = phiFac;
 
}

void 
ZRGrid::
ComputeIndexVecAndWeightsInGrid(TxVector<Standard_Real>& pos, 
				       Standard_Size indx[3], 
				       Standard_Real wl[3],
				       Standard_Real wu[3]) const
{
  Standard_Size tmpIndx[2];
  Standard_Real tmpWl[2];
  Standard_Real tmpWu[2];
  Standard_Real phiFac;

  TxVector2D<Standard_Real> tmpPos(pos[0],pos[1]);

  ComputeIndexVecAndWeightsInGrid(tmpPos,tmpIndx,tmpWl,tmpWu);

  for(Standard_Size i=0; i<2; i++){
    indx[i] = tmpIndx[i];
    wl[i] = tmpWl[i];
    wu[i] = tmpWu[i];
  }

  ComputeLocationInGridPhi(pos,indx[2],phiFac);
  wl[2] = 1.0-phiFac;
  wu[2] = phiFac;
}

void 
ZRGrid::
ComputeLocationInGridPhi(TxVector<Standard_Real>&  aLocation, 
			     Standard_Size& theIndx,
			     Standard_Real& theFrac) const
{
  Standard_Real TWOPI = 2.0*mksConsts.pi;

  Standard_Real delt_Phi = TWOPI /m_PhiNumber;
  
  theIndx = aLocation[2] / delt_Phi + m_PhiNumber ;

  theFrac = aLocation[2] / delt_Phi + m_PhiNumber -theIndx;

  theIndx = theIndx % m_PhiNumber ;
  //cout<<theIndx<<endl;
  //getchar();

  if(theIndx > m_PhiNumber-1 ||theIndx < 0)
  {

  	std::cout<<"ZRGrid::ComputeLocationInGridPhi-----------------------error"<<endl;
  }

  aLocation[2]=(theIndx+theFrac)*delt_Phi;
}

void 
ZRGrid::
ComputeFactorCrossPhi(TxVector<Standard_Real> start_Loc,Standard_Size start_index,
			TxVector<Standard_Real> end_Loc,Standard_Size end_index,
			 Standard_Real & factor_Phi)const
{

  Standard_Real PI = mksConsts.pi;
  Standard_Real delt_Phi = 2.0*mksConsts.pi /m_PhiNumber;
  bool isNeighbour = (start_index ==end_index-1)||(start_index ==end_index+1)||(start_index-end_index==m_PhiNumber-1)||(end_index-start_index==m_PhiNumber-1);
	
  if(!isNeighbour){
  	std::cout<<"ZRGrid::ComputeFactorCrossPhi------------error:start and end is: "<<start_index<<"  "<<end_index<<endl;
  	exit(1);
  }
  Standard_Size tmp_index;

  if((start_index == 0 && end_index == m_PhiNumber-1 ) ||(start_index==m_PhiNumber-1 && end_index == 0))
  {
	
	tmp_index = 0;

  }
  else{
  	 
	tmp_index = start_index > end_index? start_index:end_index;
  }
  
  //Standard_Real length_1 = fabs(start_Loc[1]*sin(tmp_index*delt_Phi-start_Loc[2]));
  //Standard_Real length_2 = fabs(end_Loc[1]*sin(tmp_index*delt_Phi-end_Loc[2]));
  //factor_Phi=length_1/(length_1+length_2);
  Standard_Real phi1 = fabs(tmp_index*delt_Phi-start_Loc[2]);
  Standard_Real phi2 = fabs(end_Loc[2]-tmp_index*delt_Phi);
  factor_Phi=phi1/(phi1+phi2);
}



