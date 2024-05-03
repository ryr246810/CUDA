#include <GridGeometry.hxx>
#include <BaseFunctionDefine.hxx>



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void  
GridGeometry::
GetAllGridFaceDatasOfPhysRgn(vector<GridFaceData*>&  theDatas) const
{

  TxSlab2D<Standard_Integer> subRgn =  m_ZRGrid->GetPhysRgn();
  GetGridFaceDatasNotOfMaterialTypeOfSubRgn( 0, subRgn, theDatas);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void  
GridGeometry::
GetAllGridFaceDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
					       vector<GridFaceData*>&  theDatas) const
{

  TxSlab2D<Standard_Integer> subRgn = m_ZRGrid->GetPhysRgn();
  GetGridFaceDatasNotOfMaterialTypesOfSubRgn( theMaterials,subRgn,theDatas);
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridGeometry::
GetGridFaceDatasNotOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					  const TxSlab2D<Standard_Integer>& subRgn,
					  vector<GridFaceData*> & theDatas) const
{
  set<Standard_Integer> theMaterials;
  theMaterials.insert(theMaterial);
  GetGridFaceDatasNotOfMaterialTypesOfSubRgn(theMaterials,subRgn,theDatas);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridGeometry::
GetGridFaceDatasNotOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
					   const TxSlab2D<Standard_Integer>& subRgn,
					   vector<GridFaceData*> & theDatas) const
{
  Standard_Size theFIndxVec[2];
  Standard_Size theFIndx;

  Standard_Integer Dir1 = 0;
  Standard_Integer Dir2 = 1;

  TxSlab2D<Standard_Integer> theRgn = m_ZRGrid->GetXtndRgn() & subRgn;
  if((theRgn.isPureZeroSpace()) || (!theRgn.isDefinedProperly())) return;

  for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<theRgn.getUpperBound(Dir1); index1++){
    theFIndxVec[Dir1] = index1;
    for(Standard_Size index2 = theRgn.getLowerBound(Dir2); index2<theRgn.getUpperBound(Dir2); index2++){
      theFIndxVec[Dir2] = index2;
      
      m_ZRGrid->FillFaceIndx(theFIndxVec, theFIndx);
      GridFace* tmpGridFace = m_Faces+theFIndx;
      
      const vector<GridFaceData*>& tmpDatas = tmpGridFace->GetFaces();
      Standard_Size nb = tmpDatas.size();
      
      for(Standard_Size j=0;j<nb;j++){
	Standard_Integer tmpMaterial = tmpDatas[j]->GetEMMaterialType();
	if(!Bit_Set_BoolOpt_AND(tmpMaterial,theMaterials)){
	  theDatas.push_back(tmpDatas[j]);
	}
      }
    } // for index2
  } // for index1
}
