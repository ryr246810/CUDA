#include <GridGeometry_Cyl3D.hxx>


GridGeometry_Cyl3D::GridGeometry_Cyl3D()
{
  m_ZRGrid = NULL;
  m_GridBndDatas = NULL;
  m_PMLDefineTool=NULL;

}


GridGeometry_Cyl3D::GridGeometry_Cyl3D(const ZRGrid* _zrgrid, const GridBndData* _bnddatas, const Standard_Size _phiDim)
{
  m_ZRGrid = _zrgrid;
  m_GridBndDatas = _bnddatas;
  m_Dimphi = _phiDim;


  m_PMLDefineTool = NULL;

  
}


GridGeometry_Cyl3D::~GridGeometry_Cyl3D()
{
  for(Standard_Size i=0;i<m_Dimphi;i++){
   delete m_Gridgeometry[i];
   }
   m_Gridgeometry.clear();

}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry_Cyl3D::GetBackGroundMaterialType() const
{
  return m_GridBndDatas->GetBackGroundMaterialType();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry_Cyl3D::GetBackGroundMaterialDataIndex() const
{
  return m_GridBndDatas->GetBackGroundMaterialDataIndex();
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry_Cyl3D::GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetMaterialTypeWithShapeIndex(theIndex);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry_Cyl3D::GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetMaterialTypeWithFaceIndex(theIndex);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer GridGeometry_Cyl3D::GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const
{
  return m_GridBndDatas->GetShapeIndexAccordingFaceIndex(theIndex);
}

