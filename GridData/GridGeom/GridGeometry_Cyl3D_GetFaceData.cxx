#include <GridGeometry_Cyl3D.hxx>
#include <BaseFunctionDefine.hxx>



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void  
GridGeometry_Cyl3D::
GetAllGridFaceDatasOfPhysRgn(vector<GridFaceData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn =  m_ZRGrid->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridFaceDatasNotOfMaterialTypeOfSubRgn( 0, subRgn, theDatas);
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void  
GridGeometry_Cyl3D::
GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					       vector<GridFaceData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = m_ZRGrid->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridFaceDatasNotOfMaterialTypesOfSubRgn( theMaterials,subRgn,theDatas);
  }
}



