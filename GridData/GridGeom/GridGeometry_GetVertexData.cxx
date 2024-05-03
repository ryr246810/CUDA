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
GetAllGridVertexDatasOfPhysRgn(const bool isExcludingAxis, 
			       vector<GridVertexData*>&  theDatas) const
{

  TxSlab2D<Standard_Integer> subRgn = GetZRGrid()->GetPhysRgn();
  GetGridVertexDatasNotOfMaterialTypeOfSubRgn(0, subRgn, isExcludingAxis,  theDatas);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void  
GridGeometry::
GetAllGridVertexDatasNotOfMaterialTypesOfPhysRgn(const set<Standard_Integer>& theMaterials, 
						 const bool isExcludingAxis, 
						 vector<GridVertexData*>&  theDatas) const
{

  TxSlab2D<Standard_Integer> subRgn = m_ZRGrid->GetPhysRgn();
  GetGridVertexDatasNotOfMaterialTypesOfSubRgn( theMaterials,subRgn, isExcludingAxis, theDatas);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridGeometry::
GetGridVertexDatasNotOfMaterialTypeOfSubRgn(const Standard_Integer theMaterial,
					    const TxSlab2D<Standard_Integer>& subRgn,
					    const bool isExcludingAxis, 
					    vector<GridVertexData*> & theDatas) const
{
  set<Standard_Integer> theMaterials;
  theMaterials.insert(theMaterial);
  GetGridVertexDatasNotOfMaterialTypesOfSubRgn(theMaterials,subRgn, isExcludingAxis, theDatas);
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
GridGeometry::
GetGridVertexDatasNotOfMaterialTypesOfSubRgn(const set<Standard_Integer>& theMaterials,
					     const TxSlab2D<Standard_Integer>& subRgn,
					     const bool isExcludingAxis, 
					     vector<GridVertexData*> & theDatas) const
{
  Standard_Size theVIndxVec[2];
  Standard_Size theVIndx;

  Standard_Integer Dir0 = 0;
  Standard_Integer Dir1 = 1;

  TxSlab2D<Standard_Integer> theRgn = m_ZRGrid->GetXtndRgn() & subRgn;
  if(!theRgn.isDefinedProperly()) return;


  for(Standard_Size index0 = theRgn.getLowerBound(Dir0); index0<=theRgn.getUpperBound(Dir0); index0++){
    theVIndxVec[Dir0] = index0;
    for(Standard_Size index1 = theRgn.getLowerBound(Dir1); index1<=theRgn.getUpperBound(Dir1); index1++){

      if(isExcludingAxis && (index1<2)) continue;

      theVIndxVec[Dir1] = index1;
      
      m_ZRGrid->FillVertexIndx(theVIndxVec, theVIndx);
      GridVertexData* tmpGridVertex = m_Vertices+theVIndx;
      
      Standard_Integer tmpMaterial = tmpGridVertex->GetEMMaterialType();
      if(!Bit_Set_BoolOpt_AND(tmpMaterial,theMaterials)){
	theDatas.push_back(tmpGridVertex);
      }
    } // for index1
  } // for index0
}
