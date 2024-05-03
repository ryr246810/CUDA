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
GetAllGridVertexDatasOfPhysRgn(const bool isExcludingAxis, 
			       vector<GridVertexData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridVertexDatasNotOfMaterialTypeOfSubRgn(0, subRgn, isExcludingAxis,  theDatas);
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
GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						 const bool isExcludingAxis, 
						 vector<GridVertexData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = m_ZRGrid->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridVertexDatasNotOfMaterialTypesOfSubRgn( theMaterials,subRgn, isExcludingAxis, theDatas);
  }
}




