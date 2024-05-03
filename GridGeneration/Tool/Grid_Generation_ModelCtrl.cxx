#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>
#include <TopTools_DataMapIteratorOfDataMapOfIntegerShape.hxx>
#include <TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger.hxx>


void Grid_Generation::BuildModelsInformation()
{
  map<Standard_Integer, Standard_Integer>* theFacesWithShape   = m_GridBndDatas->ModifyRelationBetweenFaceAndShape();
  map<Standard_Integer, vector<Standard_Integer> >* theEdgesWithFace    = m_GridBndDatas->ModifyRelationBetweenEdgeAndFace();
  map<Standard_Integer, vector<Standard_Integer> >* theVerticesWithEdge = m_GridBndDatas->ModifyRelationBetweenVertexAndEdge();


  map<Standard_Integer, Standard_Integer>* theShapesWithType  = m_GridBndDatas->ModifyShapesType();
  map<Standard_Integer, Standard_Integer>* theFacesWithType   = m_GridBndDatas->ModifyFacesType();
  map<Standard_Integer, Standard_Integer>*  theShapesWithMask = m_GridBndDatas->ModifyShapesMask();
  map<Standard_Integer, Standard_Integer>*  theFacesWithMask  = m_GridBndDatas->ModifyFacesMask();


  theFacesWithShape->clear();
  theEdgesWithFace->clear();
  theVerticesWithEdge->clear();


  theShapesWithType->clear();
  theFacesWithType->clear();
  theShapesWithMask->clear();
  theFacesWithMask->clear();


  const TColStd_DataMapOfIntegerInteger& theFacesWithShapeTool   = GetModelsCtrl()->GetFacesWithShape();
  const TColStd_DataMapOfIntegerListOfInteger& theEdgesWithFaceTool    = GetModelsCtrl()->GetEdgesWithFace();
  const TColStd_DataMapOfIntegerListOfInteger& theVerticesWithEdgeTool = GetModelsCtrl()->GetVerticesWithEdge();



  const TopTools_DataMapOfIntegerShape&  theIndexedShapesTool   = GetModelsCtrl()->GetIndexedShapes();
  const TColStd_DataMapOfIntegerInteger& theFacesWithTypeTool   = GetModelsCtrl()->GetFacesWithType();
  const TColStd_DataMapOfIntegerInteger& theFaceMaskWithIndexTool  = GetModelsCtrl()->GetFacesMask();
  const TColStd_DataMapOfIntegerInteger& theShapeMaskWithIndexTool = GetModelsCtrl()->GetShapesMask();


  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger  tmpIter1;
  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger  tmpIter2;
  TColStd_DataMapIteratorOfDataMapOfIntegerInteger tmpIter3;


  for(tmpIter1.Initialize(theEdgesWithFaceTool); tmpIter1.More(); tmpIter1.Next() ){
    const Standard_Integer theEdgeIndex = tmpIter1.Key();
    const TColStd_ListOfInteger& theFaceIndexList = tmpIter1.Value();
    vector<Standard_Integer> theFaceIndexVec; theFaceIndexVec.clear();
    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(theFaceIndexList); listIter.More(); listIter.Next() ){
      theFaceIndexVec.push_back(listIter.Value());
    }
    theEdgesWithFace->insert( pair<Standard_Integer, vector<Standard_Integer> >(theEdgeIndex, theFaceIndexVec) );
  }

  for(tmpIter2.Initialize(theVerticesWithEdgeTool); tmpIter2.More(); tmpIter2.Next() ){
    const Standard_Integer theVertexIndex = tmpIter2.Key();
    const TColStd_ListOfInteger& theEdgeIndexList = tmpIter2.Value();
    vector<Standard_Integer> theEdgeIndexVec; theEdgeIndexVec.clear();
    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(theEdgeIndexList); listIter.More(); listIter.Next() ){
      theEdgeIndexVec.push_back(listIter.Value());
    }
    theVerticesWithEdge->insert( pair<Standard_Integer, vector<Standard_Integer> >(theVertexIndex, theEdgeIndexVec) );
  }

  for(tmpIter3.Initialize(theFacesWithShapeTool); tmpIter3.More(); tmpIter3.Next() ){
    const Standard_Integer theFaceIndex = tmpIter3.Key();
    const Standard_Integer theShapeIndex = tmpIter3.Value();
    theFacesWithShape->insert( pair<Standard_Integer, Standard_Integer>(theFaceIndex, theShapeIndex) );
  }




  TopTools_DataMapIteratorOfDataMapOfIntegerShape  Iter1;
  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter2;
  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter3;
  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter4;


  for(Iter1.Initialize(theIndexedShapesTool); Iter1.More(); Iter1.Next() ){
    const Standard_Integer theIndex = Iter1.Key();
    Standard_Integer theType = GetModelsCtrl()->GetMaterialTypeWithShapeIndex(theIndex);
    theShapesWithType->insert( pair<Standard_Integer, Standard_Integer>(theIndex, theType) );
  }

  for(Iter2.Initialize(theFacesWithTypeTool); Iter2.More(); Iter2.Next() ){
    const Standard_Integer theIndex = Iter2.Key();
    const Standard_Integer theType = Iter2.Value();
    theFacesWithType->insert( pair<Standard_Integer, Standard_Integer>(theIndex, theType) );
  }

  for(Iter3.Initialize(theFaceMaskWithIndexTool); Iter3.More(); Iter3.Next() ){
    const Standard_Integer theMask = Iter3.Key();
    const Standard_Integer theIndex = Iter3.Value();
    theFacesWithMask->insert( pair<Standard_Integer, Standard_Integer>(theMask, theIndex) );
  }

  for(Iter4.Initialize(theShapeMaskWithIndexTool); Iter4.More(); Iter4.Next() ){
    const Standard_Integer theMask = Iter4.Key();
    const Standard_Integer theIndex = Iter4.Value();
    theShapesWithMask->insert( pair<Standard_Integer, Standard_Integer>(theMask, theIndex) );
  }
}

