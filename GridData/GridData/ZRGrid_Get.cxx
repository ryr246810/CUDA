#include <ZRGrid.hxx>
#include <algorithm>


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
bool 
ZRGrid::
IsIn(const Standard_Integer aDir, 
     const Standard_Real theLength) const
{
  bool tmp = true;
  if( (theLength<0) || (theLength>GetLength(aDir)) ){
    tmp = false;
  }
  return tmp;
}

bool 
ZRGrid::
IsIn(const TxVector2D<Standard_Real>& thePnt) const
{
  bool result = true;

  Standard_Integer ndim = 2;
  for(Standard_Integer dir=0; dir<ndim; dir++){
    if( (thePnt[dir]>m_RealRgn.getUpperBound(dir)) || (thePnt[dir]<m_RealRgn.getLowerBound(dir)) ){
      result = result && false;
    }else{
      result = result && true;
    }
  }

  return result;
}



Standard_Size 
ZRGrid::
GetDimension(const Standard_Integer aDir) const
{
  return m_Dimension[aDir];
}

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Size 
ZRGrid::
GetVertexDimension(const Standard_Integer aDir) const
{
  Standard_Size theDimension = m_Dimension[aDir] + 1;
  return theDimension;
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Size 
ZRGrid::
GetEdgeDimension(const Standard_Integer edgeDir, const Standard_Integer dimDir) const
{
  Standard_Size theDimension = 0;
  if(edgeDir==dimDir) theDimension = GetDimension(dimDir);
  else theDimension = GetVertexDimension(dimDir);
  return theDimension;
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Size 
ZRGrid::
GetFaceDimension(const Standard_Integer dimDir) const
{
  Standard_Size theDimension = GetDimension(dimDir);
  return theDimension;
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Size 
ZRGrid::
GetMaxIndexOfVertex(const Standard_Integer aDir) const
{
  Standard_Size result = GetVertexDimension(aDir)-1;
  return result;
}


Standard_Size 
ZRGrid::
GetMaxIndexOfEdge(const Standard_Integer aDir, const Standard_Integer dimDir) const
{
  Standard_Size result = GetEdgeDimension(aDir, dimDir)-1;
  return result;
}


Standard_Size 
ZRGrid::
GetMaxIndexOfFace(const Standard_Integer aDir) const
{
  Standard_Size result = GetFaceDimension(aDir)-1;
  return result;
}

Standard_Real 
ZRGrid::
GetStep(const Standard_Integer aDir,
	const Standard_Size anIndex) const
{
  Standard_Real theStep = m_MinSteps[aDir];

  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::const_iterator iter = m_DLVectors.find(aDir);
  if(iter != m_DLVectors.end()){
    if(anIndex<(iter->second).size()){
      theStep = (iter->second)[anIndex];
    }
  }

  return theStep;
}

// Standard_Real 
// ZRGrid::
// GetStep(const Standard_Integer aDir,
// 	const Standard_Size anIndex) const
// {
//   Standard_Real theStep = m_MinSteps[aDir];
//   // map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> > h_devDLVectors;
//   // h_devDLVectors = m_DLVectors;
//   vector<Standard_Real> vectorDLVector_z = (m_DLVectors.at(0));
//   vector<Standard_Real> vectorDLVector_r = (m_DLVectors.at(1));
//     Standard_Real *DLVPtr = NULL;
//     if(aDir == 0)
//         DLVPtr = &(vectorDLVector_z[0]);
//     else if(aDir == 1)
//         DLVPtr = &(vectorDLVector_r[0]);

//     if (DLVPtr != NULL)
//       theStep = DLVPtr[anIndex];
    
//     return theStep;
// }


TxVector2D<Standard_Real> 
ZRGrid::
GetSteps(Standard_Size indx[2]) const
{
  Standard_Integer NDIM = 2;
  TxVector2D<Standard_Real> result;
  for(Standard_Integer dir=0; dir<NDIM; dir++){
    result[dir] = GetStep(dir, indx[dir]);
  }
  return result;
}


void 
ZRGrid::
GetSteps(Standard_Size indx[2], Standard_Real result[2]) const
{
  Standard_Integer NDIM = 2;
  for(Standard_Integer dir=0; dir<NDIM; dir++){
    result[dir] = GetStep(dir, indx[dir]);
  }
}


Standard_Real 
ZRGrid::
GetDualStep(const Standard_Integer aDir, 
	    const Standard_Size indx) const
{
  Standard_Real result = 0.0;
  Standard_Size current_Index, pre_Index;

  current_Index = indx;

  if( current_Index > GetMaxIndexOfEdge(aDir,aDir) ){
    result = GetStep(aDir, GetMaxIndexOfEdge(aDir,aDir) );
  }else if(current_Index==0){
    result = GetStep(aDir, current_Index);
  }else{
    pre_Index = indx-1;
    result = 0.5*( GetStep(aDir, current_Index) + GetStep(aDir, pre_Index) );
  }

  return result;
}

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Real 
ZRGrid::
GetLength(const Standard_Integer aDir) const
{
  Standard_Size anIndex = GetMaxIndexOfVertex(aDir);
  return GetLength(aDir, anIndex);
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Real 
ZRGrid::
GetLength(const Standard_Integer aDir,
	  const Standard_Size anIndex) const
{
  Standard_Real theLength = 0.0;

  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::const_iterator iter = m_LVectors.find(aDir);
  if(iter != m_LVectors.end()){
    if(anIndex<(iter->second).size()){
      theLength = (iter->second)[anIndex];
    }
  }
  
  return theLength;
}

// Standard_Real 
// ZRGrid::
// GetLength(const Standard_Integer aDir,
// 	  const Standard_Size anIndex) const
// {
//   Standard_Real theLength = 0.0;

//   Standard_Real *LVPtr = NULL;
//     if(aDir == 0)
//         LVPtr = m_LVectors.at(0);
//     else if(aDir == 1)
//         LVPtr = m_LVectors.at(1);

//     if (LVPtr != NULL)
//       theLength = LVPtr[anIndex];
  
//   return theLength;
// }



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
Standard_Real
ZRGrid::
GetCoordComp_From_VertexVectorIndx(const Standard_Integer dir, const Standard_Size indxVec[2]) const
{
  Standard_Real result = m_Org[dir] + GetLength(dir, indxVec[dir]);
  return result;
}


Standard_Real
ZRGrid::
GetCoordComp_From_VertexScalarIndx(const Standard_Integer dir, const Standard_Size indx) const
{
  Standard_Size indxVec[2];
  FillVertexIndxVec(indx, indxVec);
  return GetCoordComp_From_VertexVectorIndx(dir, indxVec);
}


void 
ZRGrid::
GetCoord_From_VertexVectorIndx(const Standard_Size indxVec[2], Standard_Real coords[2]) const
{
  Standard_Integer NDIM = 2;
  for(Standard_Integer dir=0; dir<NDIM; dir++){
    coords[dir] = GetCoordComp_From_VertexVectorIndx(dir, indxVec);
  }
}


void 
ZRGrid::
GetCoord_From_VertexScalarIndx(const Standard_Size indx, Standard_Real coords[2]) const
{
  Standard_Size indxVec[2];
  FillVertexIndxVec(indx, indxVec);
  GetCoord_From_VertexVectorIndx( indxVec, coords);
}


TxVector2D<Standard_Real>
ZRGrid::
GetCoord_From_VertexVectorIndx(const Standard_Size indxVec[2]) const
{
  TxVector2D<Standard_Real> coords;
  Standard_Integer NDIM = 2;
  for(Standard_Integer dir=0; dir<NDIM; dir++){
    coords[dir] = GetCoordComp_From_VertexVectorIndx(dir, indxVec);
  }
  return coords;
}


TxVector2D<Standard_Real>
ZRGrid::
GetCoord_From_VertexScalarIndx(const Standard_Size indx) const
{
  Standard_Size indxVec[2];
  FillVertexIndxVec(indx, indxVec);
  return GetCoord_From_VertexVectorIndx(indxVec);
}


Standard_Size 
ZRGrid::
GetVertexSize() const
{ 
  return m_XtndVSize; 
};


Standard_Size 
ZRGrid::
GetVertexSize(const Standard_Integer aDir ) const
{
  return m_VertexSizes[aDir];
}


Standard_Size 
ZRGrid::
GetEdgeSize(Standard_Integer aDir) const
{ 
  return m_XtndESize[aDir]; 
};


Standard_Size 
ZRGrid::
GetEdgeSize(const Standard_Integer edgeDir, 
	    const Standard_Integer aDir) const
{
  return m_EdgeSizes[edgeDir][aDir];
}


Standard_Size 
ZRGrid::
GetFaceSize() const
{ 
  return m_XtndFSize; 
};


Standard_Size 
ZRGrid::
GetFaceSize(const Standard_Integer aDir) const
{
  return m_FaceSizes[aDir];
}


const TxSlab2D<Standard_Integer>& 
ZRGrid::
GetXtndRgn() const
{
  return m_XtndRgn;
}


const TxSlab2D<Standard_Integer>& 
ZRGrid::
GetPhysRgn() const
{
  return m_PhysRgn;
}


Standard_Real 
ZRGrid::
GetMinStep() const
{
  return m_MinStep;
}


Standard_Real 
ZRGrid::
GetGridLengthEpsilon() const
{
  Standard_Real result = 0.1*m_MinStep/((Standard_Real)m_Resolution);
  return result;
}


const TxSlab2D<Standard_Real>& 
ZRGrid::
GetRealRgn() const
{
  return m_RealRgn;
}

