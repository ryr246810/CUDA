#include <ZRGrid.hxx>


//----------------------------------------
//------------for vertex------------------
//----------------------------------------

void
ZRGrid::
FillVertexIndxVec(const Standard_Size indx,
		  Standard_Size indxVec[2]) const
{
  Standard_Integer ndim=2;
  Standard_Size rem = indx;
  for(Standard_Integer i=0; i<(ndim-1); ++i){
    indxVec[i] = (Standard_Size)(rem/m_VertexSizes[i]);
    rem %= m_VertexSizes[i];
  }
  indxVec[ndim-1] = rem;
}

void
ZRGrid::
FillVertexIndx(const Standard_Size indxVec[2],
	       Standard_Size& indx) const
{
  indx =
    indxVec[0]*m_VertexSizes[0]+
    indxVec[1]*m_VertexSizes[1];
}



//----------------------------------------
//------------for edge--------------------
//----------------------------------------

void
ZRGrid::
FillEdgeIndxVec(const Standard_Integer aDir,
		const Standard_Size indx,
		Standard_Size indxVec[2]) const
{
  Standard_Integer ndim=2;
  Standard_Size rem = indx;
  
  for(Standard_Integer i=0; i<(ndim-1); ++i){
    indxVec[i] = Standard_Size(rem/m_EdgeSizes[aDir][i]);
    rem %= m_EdgeSizes[aDir][i];
  }
  
  indxVec[ndim-1] = rem;
}

void
ZRGrid::
FillEdgeIndx(const Standard_Integer aDir,
	     const Standard_Size indxVec[2],
	     Standard_Size& indx) const
{
  indx =
    indxVec[0]*m_EdgeSizes[aDir][0]+
    indxVec[1]*m_EdgeSizes[aDir][1];
}



//----------------------------------------
//------------for face--------------------
//----------------------------------------

void
ZRGrid::
FillFaceIndxVec(const Standard_Size indx,
		Standard_Size indxVec[2]) const
{
  Standard_Integer ndim=2;
  Standard_Size rem = indx;
  for(Standard_Integer i=0; i<(ndim-1); ++i)
    {
      indxVec[i] = Standard_Size(rem/m_FaceSizes[i]);
      rem %= m_FaceSizes[i];
    }
  indxVec[ndim-1] = rem;
}


void
ZRGrid::
FillFaceIndx(const Standard_Size indxVec[2],
	     Standard_Size& indx) const
{
  indx =
    indxVec[0]*m_FaceSizes[0]+
    indxVec[1]*m_FaceSizes[1];
}
