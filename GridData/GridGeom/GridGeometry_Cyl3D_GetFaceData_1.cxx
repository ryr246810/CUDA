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
GetAllGridFaceDatasOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					    vector<GridFaceData*>&  theDatas) const
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  	m_Gridgeometry[i]->GetGridFaceDatasOfMaterialTypesOfSubRgn( theMaterials,subRgn,theDatas);
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
GetGridFaceDatasOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
				       const TxSlab2D<Standard_Integer>& subRgn,
				       vector<GridFaceData*> & theDatas) const
{
  set<Standard_Integer> theMaterials;
  theMaterials.insert(theMaterial);
  cout<<m_Dimphi<<endl;
  getchar();
  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
  	m_Gridgeometry[i]->GetGridFaceDatasOfMaterialTypesOfSubRgn(theMaterials,subRgn,theDatas);
  }
}


