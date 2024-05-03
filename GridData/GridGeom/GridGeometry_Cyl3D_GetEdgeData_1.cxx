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
GetAllGridEdgeDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					    const bool isExcludingAxis, 
					    vector<GridEdgeData*>& theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridEdgeDatasOfMaterialTypesOfSubRgn( theMaterials,subRgn,isExcludingAxis,theDatas);
  }
}

