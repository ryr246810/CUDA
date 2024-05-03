#include <GridGeometry.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>



void GridGeometry::BuildGridEdges()
{
  for(Standard_Integer dim =0; dim<2; dim++){
    Standard_Size nbge = m_ZRGrid->GetEdgeSize(dim);
    for(Standard_Size index=0; index<nbge; index++){
      m_Edges[dim][index].BuildEdges();
    }
  }
}


void GridGeometry::InitDefineGridEdges()
{
  m_Edges = new GridEdge*[2];

  Standard_Size e_size[2];
  Standard_Size allEdgeSize = 0;
  for(Standard_Integer i=0;i<2;i++){
    e_size[i] = m_ZRGrid->GetEdgeSize(i);
    allEdgeSize += e_size[i];
  }
  m_Edges[0] = new GridEdge[ allEdgeSize ];
  for (Standard_Size i=1;i<2;i++) m_Edges[i] = m_Edges[i-1] + e_size[i-1];

  Standard_Size indxVec[2];
  Standard_Size firstVIndx, lastVIndx;

  for(Standard_Integer dim = 0; dim <2; dim++){
    for(Standard_Size i=0; i<e_size[dim]; i++){
      m_Edges[dim][i].SetGridGeom(this);
      m_Edges[dim][i].SetDir(ZRGridLineDir(dim));
      m_Edges[dim][i].SetIndex(i);
    }
  }
}


void 
GridGeometry::
BuildGridEdgeAppendingVertices()
{
  Standard_Integer ndim = 2;
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();

  for(Standard_Integer aDir=0; aDir<ndim; aDir++){

    Standard_Integer Dir[ndim];
    for(Standard_Integer i=0; i<ndim; i++){
      Dir[i] = (aDir+i)%ndim;
    }

    Standard_Size  NE  =  GetZRGrid()->GetEdgeSize(Dir[0],Dir[0]); 

    const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData;
    theVData = m_GridBndDatas->GetEdgeBndVertexDataOf((ZRGridLineDir)(Dir[0]));

    map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >::const_iterator iter;
    for(iter = theVData->begin(); iter!= theVData->end(); iter++){
      const Standard_Size theVIndx = iter->first;

      Standard_Size theVIndxVec[ndim];
      GetZRGrid()->FillVertexIndxVec(theVIndx, theVIndxVec);

      Standard_Size theFirstGridEdgeIndx;
      GetZRGrid()->FillEdgeIndx(Dir[0], theVIndxVec, theFirstGridEdgeIndx);

      bool beInRgn = true;
      for(Standard_Integer i=1; i<ndim; i++){
	beInRgn = beInRgn && 
	  (m_ZRGrid->GetXtndRgn().getLowerBound(Dir[i]) <= theVIndxVec[Dir[i]])&&
	  (m_ZRGrid->GetXtndRgn().getUpperBound(Dir[i]) >= theVIndxVec[Dir[i]]);
      }

      if(beInRgn) {
	const vector<EdgeBndVertexData> & aVertices = iter->second;
	Standard_Size theSize = aVertices.size();

	for(Standard_Size i=0; i<theSize; i++){// for begin
	  Standard_Size index = aVertices[i].m_Index;
	  bool beInScope =
	    (m_ZRGrid->GetXtndRgn().getLowerBound(Dir[0]) <= index) && 
	    (m_ZRGrid->GetXtndRgn().getUpperBound(Dir[0]) > index);
	  
	  if(beInScope){
	    Standard_Integer aShapeIndex = aVertices[i].m_ShapeIndex;
	    Standard_Integer aFaceIndex = aVertices[i].m_FaceIndex;
	    Standard_Size aFrac = aVertices[i].m_Frac;
	    Standard_Integer aMaterialType = aVertices[i].MaterialType;

	    Standard_Integer aTransitionType = aVertices[i].TransitionType;
	    Standard_Integer aMark = BND|BNDVERTEXOFEDGE;

	    Standard_Size aGlobalEIndx = theFirstGridEdgeIndx + index*NE;
	    GridEdge* aGridEdgePointer =  m_Edges[Dir[0]] + aGlobalEIndx;

	    if(aFrac == 0 ||  aFrac == theGeomResolution){
	      DistributeEdgeBndVertex_To_SubVertex(aGridEdgePointer, 
						   aShapeIndex,
						   aFaceIndex, 
						   aFrac, 
						   aMaterialType);
	    }else{
	      AppendingVertexDataOfGridEdge* theAppendingVertex = new AppendingVertexDataOfGridEdge(aShapeIndex,
												    aFaceIndex,
												    aGridEdgePointer,
												    aFrac,
												    aMark, 
												    aMaterialType,
												    aTransitionType);

	      AddSpaceInfoOfAppendingVertexDataOfGridEdge(aGridEdgePointer, theAppendingVertex);
	      aGridEdgePointer->AddAppendingVertex(theAppendingVertex);
	    }
	  }
	}// for end
      }
    }
  }
}


void 
GridGeometry::
DistributeEdgeBndVertex_To_SubVertex(GridEdge* theEdge, 
				     Standard_Integer aShapeIndex, 
				     Standard_Integer aFaceIndex, 
				     Standard_Size aFrac, 
				     Standard_Integer aMaterialType)
{
  Standard_Integer theGeomResolution = GetZRGrid()->GetResolutionRatio();
  GridVertexData* theVertex = NULL;

  if(aFrac == 0){
    theVertex = theEdge->GetFirstVertex();
  }else if(aFrac == theGeomResolution){
    theVertex = theEdge->GetLastVertex();
  }else{
    theVertex = NULL;
  }

  if(theVertex!=NULL){
    theVertex->SetState(BND);
    RebuildfVertexDataWithNewInformations(aShapeIndex, aFaceIndex, theVertex);

    if((theVertex->GetShapeIndices()).size()==1 ){
      AddSpaceInfoOfGridVertexData(theVertex);
    }
  }
}
