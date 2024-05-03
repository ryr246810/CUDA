#include <ZRGrid.hxx>


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeBndBoxInGrid(const TxSlab2D<Standard_Real>& realRgn,
		    TxSlab2D<Standard_Size>& gridRgn) const
{
  Standard_Size rminIndex,zminIndex;
  Standard_Size rmaxIndex,zmaxIndex;

  Standard_Real zminLength = realRgn.getLowerBound(0)-GetOrg()[0];
  Standard_Real rminLength = realRgn.getLowerBound(1)-GetOrg()[1];

  Standard_Real zmaxLength = realRgn.getUpperBound(0)-GetOrg()[0];
  Standard_Real rmaxLength = realRgn.getUpperBound(1)-GetOrg()[1];

  Standard_Size aFrac;

  if(zminLength<0.0){
    zminIndex = 0;
  }else if(zminLength<GetLength(0)){
    ComputeIndex2(0,zminLength,zminIndex,aFrac);
  }else{
    zminIndex = GetVertexDimension(0)-1;
  }

  if(zmaxLength<0.0){
    zmaxIndex = 0;
  }else if(zmaxLength<GetLength(0)){
    ComputeIndex2(0,zmaxLength,zmaxIndex,aFrac);
  }else{
    zmaxIndex = GetVertexDimension(0)-1;
  }


  if(rminLength<0.0){
    rminIndex = 0;
  }else if(rminLength<GetLength(1)){
    ComputeIndex2(1,rminLength,rminIndex,aFrac);
  }else{
    rminIndex = GetVertexDimension(1)-1;
  }

  if(rmaxLength<0.0){
    rmaxIndex = 0;
  }else if(rmaxLength<GetLength(1)){
    ComputeIndex2(1,rmaxLength,rmaxIndex,aFrac);
  }else{
    rmaxIndex = GetVertexDimension(1)-1;
  }


  rmaxIndex++;
  zmaxIndex++;

  gridRgn.setBounds(zminIndex,rminIndex,zmaxIndex,rmaxIndex);
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ExtendRgnToEndAlongDir(const Standard_Integer aDir,
		       const Standard_Integer aRelativeDir,
		       TxSlab2D<Standard_Size>& gridRgn) const
{
  if(aRelativeDir==1){
    gridRgn.setUpperBound(aDir, (GetVertexDimension(aDir)-1) );
  }else{
    gridRgn.setLowerBound(aDir,0);
  }
}




/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ExtendRgnOneLayerAlongDir(const Standard_Integer aDir,
			  const Standard_Integer aRelativeDir,
			  TxSlab2D<Standard_Size>& gridRgn) const
{
  if(aRelativeDir==1){
    Standard_Real tmpIndx = gridRgn.getUpperBound(aDir) + 1;
    gridRgn.setUpperBound(aDir, tmpIndx);
  }else{
    Standard_Real tmpIndx = gridRgn.getLowerBound(aDir) - 1;
    gridRgn.setLowerBound(aDir,tmpIndx);
  }
}


