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
GetAllGridEdgeDatasOfPhysRgn(const bool isExcludingAxis, 
			     vector<GridEdgeData*>& theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn(0,subRgn,isExcludingAxis,theDatas);
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
GetAllGridEdgeDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					       const bool isExcludingAxis, 
					       vector<GridEdgeData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasNotOfMaterialTypesOfSubRgn( theMaterials,subRgn,isExcludingAxis,theDatas);
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
GetGridEdgeDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
				       const TxSlab2D<Standard_Integer>& subRgn,
				       const bool isExcludingAxis, 
				       vector<GridEdgeData*> & theDatas) const
{
  set<Standard_Integer> theMaterials;
  theMaterials.insert(theMaterial);
  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	m_Gridgeometry[i]->GetGridEdgeDatasOfMaterialTypesOfSubRgn(theMaterials,subRgn,isExcludingAxis,theDatas);
  }
}


