#include <GridGeometry.hxx>

#include <AppendingVertexDataOfGridEdge.hxx>

void GridGeometry::InitDefineGridVertices()
{
  Standard_Size v_size = m_ZRGrid->GetVertexSize();
  m_Vertices = new GridVertexData[v_size];

  Standard_Integer backGroundMatType = GetBackGroundMaterialType();
  Standard_Integer backGroundMatDataIndex = GetBackGroundMaterialDataIndex();

  for(Standard_Size index=0; index<v_size; index++ ){
    m_Vertices[index].SetGridGeom(this);
    m_Vertices[index].SetState(OUTSHAPE);
    m_Vertices[index].SetType(REGVERTEX);
    m_Vertices[index].SetMaterialType( backGroundMatType );
    if(backGroundMatType==USERDEFINED){
      m_Vertices[index].SetMaterialDataIndex( backGroundMatDataIndex );
    }
    m_Vertices[index].SetIndex(index);
  }
}


void
GridGeometry::
SetupGridVertices()
{
  Standard_Size v_size = m_ZRGrid->GetVertexSize();
  for(Standard_Size index=0; index<v_size; index++ ){
    m_Vertices[index].Setup();
  }
}


void
GridGeometry::
BuildGridVertices()
{
  BuildGridVertices_With_EdgeBnd_Along(0);  // zdir
  BuildGridVertices_With_EdgeBnd_Along(1);  // rdir
}


void GridGeometry::BuildSurroundingGeomElements()
{
  Standard_Size v_size = m_ZRGrid->GetVertexSize();
  for(Standard_Size index=0; index<v_size; index++ ){
    m_Vertices[index].BuildSharingGridFaceDatas();
    m_Vertices[index].BuildDivTEdges();
    m_Vertices[index].BuildSharedTDFaces();
  }
}


void
GridGeometry::
BuildGridVertices_With_EdgeBnd_Along(const Standard_Integer aDir)
{
  // if a GridEdge Contain an AppendingVertex |---------------+--------------|,
  //                                      FirstGV        AppendingVertex   LastGV
  //     then the FirstGVIndex is the LastGV's Index of the GridEdge
  
  Standard_Integer ndim = 2;

  Standard_Integer dir0 = aDir;
  Standard_Integer dir1 = (aDir+1)%2;

  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData = m_GridBndDatas->GetEdgeBndVertexDataOf( (ZRGridLineDir)(dir0) );

  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >::const_iterator iter;

  Standard_Size NV = GetZRGrid()->GetVertexSize(dir0);

  for(iter = theVData->begin(); iter!= theVData->end(); iter++){
    Standard_Size theIndx = iter->first;
    Standard_Size theIndxVec[ndim];
    
    GetZRGrid()->FillVertexIndxVec(theIndx, theIndxVec);
    bool beInRgn = (m_ZRGrid->GetXtndRgn().getLowerBound(dir1) <= theIndxVec[dir1]) && (m_ZRGrid->GetXtndRgn().getUpperBound(dir1) >= theIndxVec[dir1]);
    
    beInRgn = true;

    if(beInRgn) {
      const vector<EdgeBndVertexData> & theBndVertives = iter->second;
      Standard_Size theSize = theBndVertives.size();

      if(dir0==1){  // rdirection
	if(theSize>0){
	  GridVertexData* theFirstGV = m_Vertices + theIndx;

	  //if( (theFirstGV->HasShapeIndex(theBndVertives[0].m_ShapeIndex)) && (theBndVertives[0].TransitionType==-1) ) {
	  if( theBndVertives[0].TransitionType==-1 ) {
	    Standard_Integer theShapeIndex = theBndVertives[0].m_ShapeIndex;
	    Standard_Size theFirstGVIndex = 0;
	    Standard_Size theLastGVIndex  = theBndVertives[0].m_Index; 
	    SetGridVertices_AsInShape_InScope(dir0, theIndx, theFirstGVIndex, theLastGVIndex, theShapeIndex);
	  }
	}
      }

      if(theSize>1){
	for(Standard_Size i=0; i<(theSize-1); i++){
	  if(theBndVertives[i].m_ShapeIndex == theBndVertives[i+1].m_ShapeIndex){ //the IntPnts of one same shape
	    if(theBndVertives[i].TransitionType==1 && theBndVertives[i+1].TransitionType==-1 ){
	      Standard_Integer theShapeIndex = theBndVertives[i].m_ShapeIndex;
	      
	      Standard_Size theFirstGVIndex =  theBndVertives[i].m_Index+1;
	      Standard_Size theLastGVIndex  = theBndVertives[i+1].m_Index;
	      
	      SetGridVertices_AsInShape_InScope(dir0,theIndx, theFirstGVIndex, theLastGVIndex, theShapeIndex);
	    }
	  }
	} // for(Standard_Size i=0; i<(theSize-1); i++)
      }// if(theSize>1) 
    }
  }
}


void
GridGeometry::
SetGridVertices_AsInShape_InScope(const Standard_Integer aDir,
				  const Standard_Size theIndx,
				  const Standard_Size theFirstGVIndex, 
				  const Standard_Size theLastGVIndex,  
				  const Standard_Integer theShapeIndex)
{
  Standard_Size NV = GetZRGrid()->GetVertexSize(aDir);

  Standard_Size theLocalLowerBnd = m_ZRGrid->GetXtndRgn().getLowerBound(aDir);
  Standard_Size theLocalUpperBnd = m_ZRGrid->GetXtndRgn().getUpperBound(aDir);

  for(Standard_Size j = theFirstGVIndex; j<= theLastGVIndex; j++){
    if( (theLocalLowerBnd <= j) && (j <= theLocalUpperBnd) ){
      Standard_Size aLocalVIndx  = theIndx + j*NV;
      m_Vertices[aLocalVIndx].SetState(INSHAPE);
      
      Standard_Size aLocalVIndxVec[2];
      GetZRGrid()->FillVertexIndxVec(aLocalVIndx, aLocalVIndxVec);
      m_Vertices[aLocalVIndx].AddShapeIndex(theShapeIndex);
      const set<Standard_Integer>& theIndices = m_Vertices[aLocalVIndx].GetShapeIndices();
      Standard_Integer theMaterialType = m_GridBndDatas->GetMaterialTypeWithShapeIndices(theIndices); 
      m_Vertices[aLocalVIndx].SetMaterialType(theMaterialType); // set as previous material type
      
      // remove the material datas of space defination
      const vector<Standard_Integer>& theDatas = m_GridBndDatas->GetMatDataIndicesOfSpaceDefine();
      m_Vertices[aLocalVIndx].RemoveMatDataIndices(theDatas);
 
      // be set as only one material data index
      Standard_Integer theMatDataIndex;
      if(GetGridBndDatas()->HasShapeMaterialDataIndex(theShapeIndex, theMatDataIndex)){
	m_Vertices[aLocalVIndx].AppendMatDataIndex(theMatDataIndex);
      }
    }
  }
}



void 
GridGeometry::
RemoveSpaceInfoOfAppendingVertexDataOfGridEdge(const GridEdge* theEdge, AppendingVertexDataOfGridEdge* theVertex)
{
  Standard_Integer theMaterialType;
  set<Standard_Integer> theMaterialIndices;
  GetMaterialOfGridEdgeOnlyAccordingSpaceDefine(theEdge, theMaterialType, theMaterialIndices);

  theVertex->DelMaterialType(theMaterialType);
  theVertex->RemoveMatDataIndices(theMaterialIndices);
}



void 
GridGeometry::
GetMaterialOfGridEdgeOnlyAccordingSpaceDefine(const GridEdge* theEdge, 
					      Standard_Integer& theMaterialType, 
					      set<Standard_Integer>& theMaterialIndices)
{
  GridVertexData* firstVertex = theEdge->GetFirstVertex();
  GridVertexData* lastVertex = theEdge->GetLastVertex();

  Standard_Integer firstMaterialType;
  set<Standard_Integer> firstMaterialIndices;
  GetMaterialOfGridVertexDataOnlyAccordingSpaceDefine(firstVertex, firstMaterialType, firstMaterialIndices);

  Standard_Integer lastMaterialType;
  set<Standard_Integer> lastMaterialIndices;
  GetMaterialOfGridVertexDataOnlyAccordingSpaceDefine(lastVertex, lastMaterialType, lastMaterialIndices);


  theMaterialType = firstMaterialType | lastMaterialType;


  theMaterialIndices.clear();
  set<Standard_Integer>::const_iterator firstIter;
  for(firstIter = firstMaterialIndices.begin(); firstIter!=firstMaterialIndices.end(); firstIter++) {
    Standard_Integer currMatDataIndex = *firstIter;
    set<Standard_Integer>::iterator iter = theMaterialIndices.find(currMatDataIndex);
    if(iter==theMaterialIndices.end()){
      theMaterialIndices.insert(currMatDataIndex);
    }
  }

  set<Standard_Integer>::const_iterator lastIter;
  for(lastIter = lastMaterialIndices.begin(); lastIter!=lastMaterialIndices.end(); lastIter++) {
    Standard_Integer currMatDataIndex = *lastIter;
    set<Standard_Integer>::iterator iter = theMaterialIndices.find(currMatDataIndex);
    if(iter==theMaterialIndices.end()){
      theMaterialIndices.insert(currMatDataIndex);
    }
  }
}


void 
GridGeometry::
GetMaterialOfGridVertexDataOnlyAccordingSpaceDefine(const GridVertexData* theVertex, 
						    Standard_Integer& theMaterialType, 
						    set<Standard_Integer>& theMaterialIndices)
{
  Standard_Integer backGroundMatType = GetBackGroundMaterialType();
  Standard_Integer backGroundMatDataIndex = GetBackGroundMaterialDataIndex();

  theMaterialType = backGroundMatType;

  theMaterialIndices.clear();
  if(backGroundMatType==USERDEFINED){
    theMaterialIndices.insert(backGroundMatDataIndex);
  }
}



void 
GridGeometry::
AddSpaceInfoOfAppendingVertexDataOfGridEdge(const GridEdge* theEdge, AppendingVertexDataOfGridEdge* theVertex)
{
  Standard_Integer theMaterialType;
  set<Standard_Integer> theMaterialIndices;
  GetMaterialOfGridEdgeOnlyAccordingSpaceDefine(theEdge, theMaterialType, theMaterialIndices);

  theVertex->AddMaterialType(theMaterialType);
  theVertex->AppendMatDataIndices(theMaterialIndices);
}




void 
GridGeometry::
RebuildfVertexDataWithNewInformations(const Standard_Integer theShapeIndex, 
				      const Standard_Integer theFaceIndex, 
				      VertexData* theData)
{
  theData->AddShapeIndex(theShapeIndex);
  theData->AddFaceIndex(theFaceIndex);

  const set<Standard_Integer>& theAllIndices = theData->GetShapeIndices();
  Standard_Integer theMaterialType = m_GridBndDatas->GetMaterialTypeWithShapeIndices(theAllIndices); //  include the new added shape material
  theData->SetMaterialType(theMaterialType);
  
  const set<Standard_Integer>& theAllFaceIndices = theData->GetFaceIndices();
  Standard_Integer theFaceMaterialType = m_GridBndDatas->GetMaterialTypeWithFaceIndices(theAllFaceIndices); // include the new added face material
  theData->AddMaterialType(theFaceMaterialType); 
  
  const vector<Standard_Integer>& theDatas = m_GridBndDatas->GetMatDataIndicesOfSpaceDefine();
  theData->RemoveMatDataIndices(theDatas);
  
  Standard_Integer theMatDataIndex;
  if(GetGridBndDatas()->HasShapeMaterialDataIndex(theShapeIndex, theMatDataIndex)){
    theData->AppendMatDataIndex(theMatDataIndex);
  }
}



void 
GridGeometry::
RebuildfVertexDataWithNewInformations(const Standard_Integer theShapeIndex, 
				      const Standard_Integer theEdgeIndex, 
				      const set<Standard_Integer>& theFaceIndices, 
				      VertexData* theData)
{
  theData->AddShapeIndex(theShapeIndex);
  theData->AddEdgeIndex(theEdgeIndex);
  theData->AddFaceIndices(theFaceIndices);

  const set<Standard_Integer>& theAllIndices = theData->GetShapeIndices();
  Standard_Integer theMaterialType = m_GridBndDatas->GetMaterialTypeWithShapeIndices(theAllIndices); // include the new added shape material
  theData->SetMaterialType(theMaterialType); 
  
  const set<Standard_Integer>& theAllFaceIndices = theData->GetFaceIndices();
  Standard_Integer theFaceMaterialType = m_GridBndDatas->GetMaterialTypeWithFaceIndices(theAllFaceIndices); // include the new added face material
  theData->AddMaterialType(theFaceMaterialType); 
  
  const vector<Standard_Integer>& theDatas = m_GridBndDatas->GetMatDataIndicesOfSpaceDefine();
  theData->RemoveMatDataIndices(theDatas);
  
  Standard_Integer theMatDataIndex;
  if(GetGridBndDatas()->HasShapeMaterialDataIndex(theShapeIndex, theMatDataIndex)){
    theData->AppendMatDataIndex(theMatDataIndex);
  }
}



void 
GridGeometry::
AddSpaceInfoOfGridVertexData(GridVertexData* theVertex)
{
  Standard_Integer theMaterialType;
  set<Standard_Integer> theMaterialIndices;
  GetMaterialOfGridVertexDataOnlyAccordingSpaceDefine(theVertex, theMaterialType, theMaterialIndices);

  theVertex->AddMaterialType(theMaterialType);
  theVertex->AppendMatDataIndices(theMaterialIndices);
}
