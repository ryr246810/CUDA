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
GetAllGridEdgeDatasNotOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials, 
					       vector<GridEdgeData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasNotOfMaterialTypesAlongAxis( theMaterials,subRgn,theDatas);
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
GetAllGridEdgeDatasOfMaterialTypesAlongAxis(const set<Standard_Integer>& theMaterials, 
					    vector<GridEdgeData*>& theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasOfMaterialTypesAlongAxis( theMaterials,subRgn,theDatas);
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
GetAllGridEdgeDatasAlongAxis(vector<GridEdgeData*>& theDatas) const
{

  //if(m_Dimphi < 2) return;
  set<Standard_Integer> theMaterials;
  theMaterials.insert(0);
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasNotOfMaterialTypesAlongAxis(theMaterials,subRgn,theDatas);
  }
}

