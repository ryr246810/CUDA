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
GetAllGridVertexDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					      const bool isExcludingAxis, 
					      vector<GridVertexData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridVertexDatasOfMaterialTypesOfSubRgn( theMaterials,subRgn,isExcludingAxis,theDatas);
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
GetGridVertexDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					 const TxSlab2D<Standard_Integer>& subRgn,
					 const bool isExcludingAxis, 
					 vector<GridVertexData*> & theDatas) const
{
  set<Standard_Integer> theMaterials;
  theMaterials.insert(theMaterial);

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	m_Gridgeometry[i]->GetGridVertexDatasOfMaterialTypesOfSubRgn(theMaterials,subRgn,isExcludingAxis,theDatas);
  }
}


