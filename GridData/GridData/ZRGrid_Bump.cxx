#include <ZRGrid.hxx>


Standard_Size
ZRGrid::
bumpFace(const Standard_Size dir,
	 const Standard_Size indx,
	 const Standard_Size amt) const
{
  Standard_Size tmp = indx + m_FaceSizes[dir] *amt;
  return tmp;
}

Standard_Size
ZRGrid::
bumpFace(const Standard_Size dir,
	 const Standard_Size indx) const
{

  return bumpFace(dir,indx,1);
}

Standard_Size
ZRGrid::
bumpVertex(const Standard_Size dir,
	   const Standard_Size indx,
	   const Standard_Size amt) const
{
  Standard_Size tmp = indx + m_VertexSizes[dir] *amt;
  return tmp;

}


Standard_Size
ZRGrid::
bumpVertex(const Standard_Size dir,
	   const Standard_Size indx) const
{

  return bumpVertex(dir,indx,1);
}


Standard_Size
ZRGrid::
bumpEdge(const Standard_Size edgedir,
	 const Standard_Size bumpdir,
	 const Standard_Size indx,
	 const Standard_Size amt) const
{
  Standard_Size tmp = indx + m_EdgeSizes[edgedir][bumpdir] *amt;
  return tmp;
  
}


Standard_Size
ZRGrid::
bumpEdge(const Standard_Size edgedir,
	 const Standard_Size bumpdir,
	 const Standard_Size indx) const
{
  return bumpEdge(edgedir,bumpdir,indx,1);
}



Standard_Size 
ZRGrid::
iBumpFace(const Standard_Size dir,
	  const Standard_Size indx,
	  const Standard_Size amt) const
{
  Standard_Size tmp = indx - m_FaceSizes[dir] *amt;
  return tmp;
}

Standard_Size
ZRGrid::
iBumpFace(const Standard_Size dir,
	  const Standard_Size indx) const
{
  return iBumpFace(dir,indx,1);
}

Standard_Size
ZRGrid::
iBumpVertex(const Standard_Size dir,
	    const Standard_Size indx,
	    const Standard_Size amt) const
{
  Standard_Size tmp = indx - m_VertexSizes[dir] *amt;
  return tmp;
}

Standard_Size
ZRGrid::
iBumpVertex(const Standard_Size dir,
	    const Standard_Size indx) const
{
  return iBumpVertex(dir,indx,1);
}

Standard_Size
ZRGrid::
iBumpEdge(const Standard_Size edgedir,
	  const Standard_Size bumpdir,
	  const Standard_Size indx,
	  const Standard_Size amt) const
{
  Standard_Size tmp = indx - m_EdgeSizes[edgedir][bumpdir] *amt;
  return tmp;
}

Standard_Size
ZRGrid::
iBumpEdge(const Standard_Size edgedir,
	  const Standard_Size bumpdir,
	  const Standard_Size indx) const
{
  return iBumpEdge(edgedir,bumpdir,indx,1);
}




/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/*********************************************************************/
/******************************* bumpto ******************************/

Standard_Size 
ZRGrid::
bumpVertexto(const Standard_Size theVertexIndex,
	     const Standard_Size bumpdir,
	     const Standard_Size bumpLocation) const
{
  Standard_Size result;
  Standard_Size theVecIndex[2];
  FillVertexIndxVec(theVertexIndex, theVecIndex);
  theVecIndex[bumpdir] = bumpLocation;
  FillVertexIndx(theVecIndex, result);
  return result;
}


Standard_Size 
ZRGrid::
bumpEdgeto(const Standard_Size theEdgeIndex,
	   const Standard_Size edgedir,
	   const Standard_Size bumpdir,
	   const Standard_Size bumpLocation) const
{
  Standard_Size result;
  Standard_Size theVecIndex[2];
  FillEdgeIndxVec(edgedir, theEdgeIndex, theVecIndex);
  theVecIndex[bumpdir] = bumpLocation;
  FillEdgeIndx(edgedir, theVecIndex, result);
  return result;
}


Standard_Size 
ZRGrid::
bumpFaceto(const Standard_Size theFaceIndex,
	   const Standard_Size bumpdir,
	   const Standard_Size bumpLocation) const
{
  Standard_Size result;
  Standard_Size theVecIndex[2];
  FillFaceIndxVec(theFaceIndex, theVecIndex);
  theVecIndex[bumpdir] = bumpLocation;
  FillFaceIndx(theVecIndex, result);
  return result;
}

