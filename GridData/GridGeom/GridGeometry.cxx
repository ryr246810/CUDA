#include <GridGeometry.hxx>


GridGeometry::GridGeometry()
{
  m_ZRGrid = NULL;
  m_GridBndDatas = NULL;
  m_PMLDefineTool=NULL;

  m_Vertices=NULL;
  m_Edges=NULL;
  m_Faces=NULL;

  m_PhiIndex  = -1 ;
  m_PhiNumber = 1 ;

}


GridGeometry::GridGeometry(const ZRGrid* _zrgrid, const GridBndData* _bnddatas)
{
  m_ZRGrid = _zrgrid;
  m_GridBndDatas = _bnddatas;
  m_PMLDefineTool = NULL;

  m_Vertices=NULL;
  m_Edges=NULL;
  m_Faces=NULL;

  m_PhiIndex  = -1 ;
  m_PhiNumber = 1 ;
}


GridGeometry::~GridGeometry()
{
  if(m_Faces!=NULL){
    delete[] m_Faces;
  }

  if(m_Edges!=NULL) {
    delete[] m_Edges[0];
    delete[] m_Edges;
  }

  if(m_Vertices!=NULL){
    delete[] m_Vertices;
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry::GetBackGroundMaterialType() const
{
  return m_GridBndDatas->GetBackGroundMaterialType();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry::GetBackGroundMaterialDataIndex() const
{
  return m_GridBndDatas->GetBackGroundMaterialDataIndex();
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry::GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetMaterialTypeWithShapeIndex(theIndex);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry::GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetMaterialTypeWithFaceIndex(theIndex);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry::GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetShapeIndexAccordingFaceIndex(theIndex);
}

