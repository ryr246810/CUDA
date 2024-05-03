#include <GridGeometry.hxx>
#include <T_Element.hxx>

#include <AppendingVertexDataOfGridFace.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>


/*****************************************************/
// Function: SetupGridFaces
// Purpose:  memory locating for GridFace
/****************************************************/
void GridGeometry::InitDefineGridFaces()
{
  Standard_Size f_size = m_ZRGrid->GetFaceSize();
  m_Faces = new GridFace[ f_size ];

  for(Standard_Size i=0; i<f_size; i++){
    m_Faces[i].SetGridGeom(this);
    m_Faces[i].SetIndex(i);
  }
}



/*****************************************************/
// Function:
// Purpose:
/****************************************************/
void GridGeometry::BuildGridFaceDatas()
{
  Standard_Size FacesNum = m_ZRGrid->GetFaceSize();
  for(Standard_Size indx=0; indx<FacesNum; indx++){
    GridFace* aFacePointer = m_Faces + indx;
    aFacePointer->BuildFaces();
  }
}



/*****************************************************/
// Function:
// Purpose:
/****************************************************/
void GridGeometry::BuildGridFaceAppendingVertices()
{
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();

  Standard_Integer Dir1 = 0;
  Standard_Integer Dir2 = 1;
  
  const vector<FaceBndVertexData>& aCVertices = *m_GridBndDatas->GetFaceBndVertexData();
  Standard_Size theSize = aCVertices.size();
  
  for(Standard_Size i=0; i<theSize; i++){// for begin
    
    Standard_Size theFaceGlobalIndx = aCVertices[i].m_Index;
    Standard_Integer theShapeIndex = aCVertices[i].m_ShapeIndex;
    Standard_Integer theEdgeIndex = aCVertices[i].m_EdgeIndex;
    
    set<Standard_Integer> theFaceIndices;
    m_GridBndDatas->GetFaceIndices_With_EdgeIndex(theEdgeIndex, theFaceIndices);
    
    Standard_Size aFrac1 = aCVertices[i].m_Frac1;
    Standard_Size aFrac2 = aCVertices[i].m_Frac2;
    Standard_Integer aMaterialType = aCVertices[i].MaterialType;
    
    Standard_Size theFaceGlobalIndxVec[2];
    m_ZRGrid->FillFaceIndxVec(theFaceGlobalIndx, theFaceGlobalIndxVec);
    
    if( (m_ZRGrid->GetXtndRgn().getLowerBound(Dir1) <= theFaceGlobalIndxVec[Dir1]) && 
	(theFaceGlobalIndxVec[Dir1] <  m_ZRGrid->GetXtndRgn().getUpperBound(Dir1)) &&
	(m_ZRGrid->GetXtndRgn().getLowerBound(Dir2) <= theFaceGlobalIndxVec[Dir2]) && 
	(theFaceGlobalIndxVec[Dir2] <  m_ZRGrid->GetXtndRgn().getUpperBound(Dir2)) ){
      
      GridFace* aBaseFace =  m_Faces + theFaceGlobalIndx;
      
      if( (aFrac1==0) || (aFrac1==theGeomResolution) || 
	  (aFrac2==0) || (aFrac2==theGeomResolution) ) {
	DistributeFaceBndVertex_To_SubElement(aBaseFace, 
					      theShapeIndex, 
					      theEdgeIndex, 
					      theFaceIndices, 
					      aFrac1,
					      aFrac2, 
					      aMaterialType);
      }else{
	Standard_Integer aMark = BND|BNDVERTEXOFFACE;
	AppendingVertexDataOfGridFace* theFaceBndVertex = new AppendingVertexDataOfGridFace(theShapeIndex, 
											    theEdgeIndex,
											    theFaceIndices,
											    aBaseFace,
											    aFrac1,
											    aFrac2, 
											    aMark,
											    aMaterialType);
	aBaseFace->AddBndVertexOfGridFace(theFaceBndVertex);
      }
    }
  }// for end 
}



/*****************************************************/
// Function:
// Purpose:
/****************************************************/
void 
GridGeometry::
DistributeFaceBndVertex_To_SubElement(GridFace* theFace, 
				      Standard_Integer theShapeIndex, 
				      Standard_Integer theEdgeIndex, 
				      const set<Standard_Integer>& theFaceIndices, 
				      Standard_Size aFrac1, 
				      Standard_Size aFrac2, 
				      Standard_Integer aMaterialType)
{
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();

  Standard_Integer theRuleVec[2];

  theRuleVec[0]=0;
  theRuleVec[1]=0;
 
  if(aFrac1==0 || aFrac1 == theGeomResolution){
    theRuleVec[0] = 1;
  }
  if(aFrac2==0 || aFrac2 == theGeomResolution){
    theRuleVec[1] = 1;
  }

  Standard_Integer theRule = theRuleVec[0] + theRuleVec[1];

  if(theRule == 1){
    DistributeFaceBndVertex_To_SubEdge(theFace, 
				       theShapeIndex, 
				       theEdgeIndex, 
				       theFaceIndices, 
				       aFrac1, aFrac2, 
				       aMaterialType);
  }else if(theRule == 2){
    DistributeFaceBndVertex_To_SubVertex(theFace, 
					 theShapeIndex, 
					 theEdgeIndex, 
					 theFaceIndices, 
					 aFrac1, aFrac2, 
					 aMaterialType);
  }else{
    cout<<"GridGeometry::DistributeFaceBndVertex_To_SubElement---------------------------------error"<<endl;
  }
}



/*****************************************************/
// Function:
// Purpose:
/****************************************************/
void
GridGeometry::
DistributeFaceBndVertex_To_SubEdge(GridFace* theFace, 
				   Standard_Integer theShapeIndex, 
				   Standard_Integer theEdgeIndex, 
				   const set<Standard_Integer>& theFaceIndices, 
				   Standard_Size aFrac1, 
				   Standard_Size aFrac2, 
				   Standard_Integer aMaterialType)
{
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();

  Standard_Integer FDir1 = 0;
  Standard_Integer FDir2 = 1;

  Standard_Size indxVec[2];
  theFace->GetVecIndex(indxVec);

  Standard_Integer theEdgeDir;

  Standard_Integer dim[2];
  for(Standard_Size i=0; i<2; i++){
    dim[i] = 0;
  }

  Standard_Size theFrac = 0;
  if(aFrac1 == 0 || aFrac1 == theGeomResolution){
    theFrac = aFrac2;
    theEdgeDir = FDir2;

    if(aFrac1 == theGeomResolution){
      dim[FDir1] = 1;
    }
  }else if(aFrac2 == 0 || aFrac2 == theGeomResolution){
    theFrac = aFrac1;
    theEdgeDir = FDir1;

    if(aFrac2 == theGeomResolution){
      dim[FDir2] = 1;
    }
  }else{
    cout<<"GridGeometry::DistributeFaceBndVertex_To_SubEdge---------------------------------error----------1"<<endl;
  }

  for(Standard_Size i=0; i<2; i++){
    indxVec[i]+=dim[i];
  }
  Standard_Size theEdgeScalarIndx;
  GetZRGrid()->FillEdgeIndx(theEdgeDir, indxVec, theEdgeScalarIndx);

  GridEdge* theEdge = m_Edges[theEdgeDir] + theEdgeScalarIndx;

  vector<AppendingVertexDataOfGridEdge*>& theEdgeBndVertices = theEdge->ModifyAppendingVertices();
  if(!theEdgeBndVertices.empty()){
    vector<AppendingVertexDataOfGridEdge*>::iterator iter;
    for(iter = theEdgeBndVertices.begin(); iter!=theEdgeBndVertices.end(); iter++){
      AppendingVertexDataOfGridEdge* currBndVertex = *iter;

      if(  currBndVertex->HasCommonFaceIndexWith(theFaceIndices) ){
	if(currBndVertex->GetFrac() == theFrac){
	  RebuildfVertexDataWithNewInformations(theShapeIndex, theEdgeIndex, theFaceIndices, currBndVertex);
	  if((currBndVertex->GetShapeIndices()).size()==1 ){
	    AddSpaceInfoOfAppendingVertexDataOfGridEdge(theEdge, currBndVertex);
	  }
	}else{
	  cout<<"GridGeometry::DistributeFaceBndVertex_To_SubEdge---------------------------------error----------2"<<endl;
	}
      }else{
	cout<<"GridGeometry::DistributeFaceBndVertex_To_SubEdge---------------------------------error----------3"<<endl;
      }
    }
  }else{
    cout<<"GridGeometry::DistributeFaceBndVertex_To_SubEdge---------------------------------error----------4"<<endl;
  }
}




void 
GridGeometry::
DistributeFaceBndVertex_To_SubVertex(GridFace* theFace, 
				     Standard_Integer theShapeIndex, 
				     Standard_Integer theEdgeIndex, 
				     const set<Standard_Integer>& theFaceIndices, 
				     Standard_Size aFrac1, 
				     Standard_Size aFrac2, 
				     Standard_Integer aMaterialType)
{
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();

  Standard_Integer FDir1 = 0;
  Standard_Integer FDir2 = 1;

  Standard_Size indxVec[2];
  theFace->GetVecIndex(indxVec);

  Standard_Integer theEdgeDir;

  Standard_Integer dim[2];
  for(Standard_Size i=0; i<2; i++){
    dim[i] = 0;
  }

  if(aFrac1 == theGeomResolution){
    dim[FDir1] += 1;
  }else if(aFrac2 == theGeomResolution){
    dim[FDir2] += 1;
  }else{
  }

  for(Standard_Size i=0; i<2; i++){
    indxVec[i]+=dim[i];
  }

  Standard_Size theVertexIndx;
  GetZRGrid()->FillVertexIndx(indxVec, theVertexIndx);
  GridVertexData* theVertex = m_Vertices + theVertexIndx;

  theVertex->SetState(BND);

  RebuildfVertexDataWithNewInformations(theShapeIndex, theEdgeIndex, theFaceIndices, theVertex);

  if((theVertex->GetShapeIndices()).size()==1 ){
    AddSpaceInfoOfGridVertexData(theVertex); 
  }
}

