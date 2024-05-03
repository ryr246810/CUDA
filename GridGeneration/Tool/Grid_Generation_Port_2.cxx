#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>

//#define MESH5_DBG


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::ExtendEdgeBndVerticesInRgn(const Standard_Integer thePortIndex,
						 const ZRGridLineDir& theDir, 
						 const Standard_Integer theRelativeDir,
						 const TxSlab2D<Standard_Size>& theRgn)
{
  std::set<Standard_Integer> theNBFaceIndices;
  GetModelsCtrl()->FindNeighBourFacesOfOneFace(thePortIndex, theNBFaceIndices);
  if(theNBFaceIndices.empty()){
    cout<<"Grid_Generation::ExtendEdgeBndVerticesInRgn------------error---------portindex"<<endl;
    return;
  }

  Standard_Integer Dir0 = Standard_Integer(theDir);
  Standard_Integer Dir1 = (Dir0+1)%2;

  Standard_Size Index0;
  if(theRelativeDir==1) Index0 = theRgn.getLowerBound(Dir0);
  else                  Index0 = theRgn.getUpperBound(Dir0);

  Standard_Size  NMIN0,NMAX0,NMIN1,NMAX1;
  Standard_Size  SIZE0;

  NMIN0 = theRgn.getLowerBound(Dir0);
  NMAX0 = theRgn.getUpperBound(Dir0);

  NMIN1 = theRgn.getLowerBound(Dir1);
  NMAX1 = theRgn.getUpperBound(Dir1);

  SIZE0 = GetZRGrid()->GetVertexSize( Standard_Integer(Dir0) );

#ifdef MESH5_DBG
  cout<<"Grid_Generation::ExtendEdgeBndVerticesInRgn--------------------1"<<endl;
  theRgn.write(cout);
  cout<<"theDir = "<<theDir<<endl;
  cout<<"Index0 = "<<Index0<<endl;
  cout<<"NMIN0 = "<<NMIN0<<"\t"<<"NMAX0 = "<<NMAX0<<endl;
  cout<<"NMIN1 = "<<NMIN1<<"\t"<<"NMAX1 = "<<NMAX1<<endl;
  cout<<"Grid_Generation::ExtendEdgeBndVerticesInRgn--------------------2"<<endl;
#endif

  Standard_Size theIndex = Index0*SIZE0;
  
  if(m_GridBndDatas->HasEdgeBndVertexDataOf(ZRGridLineDir(Dir1), theIndex)){
    const vector<EdgeBndVertexData>& theVertexDatas = m_GridBndDatas->GetEdgeBndVertexDataOf(ZRGridLineDir(Dir1), theIndex);
    Standard_Size nb=theVertexDatas.size();
    for(Standard_Size i=0;i<nb;i++){
      if( NMIN1<=(theVertexDatas[i].m_Index) && (theVertexDatas[i].m_Index)<NMAX1 ){
	std::set<Standard_Integer>::iterator iter = theNBFaceIndices.find(theVertexDatas[i].m_FaceIndex);
	if(iter!=theNBFaceIndices.end()){
	  EdgeBndVertexData aBndVertex;
	  CopyEdgeBndVertexFrom(theVertexDatas[i], aBndVertex);

	  // be careful: avoid repetivly to set the value of Index0 according Index0; 
	  for(Standard_Integer targetIndex = NMIN0; targetIndex<=NMAX0; targetIndex++){
	    if(targetIndex==Index0) continue;
	    Standard_Size theRayIndex = targetIndex*SIZE0;
	    InsertEdgeBndVertexData(ZRGridLineDir(Dir1), theRayIndex, aBndVertex);
	  }

	}
      }
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  :  Insert EdgeBndVertexData into the "sequence" of the BndVertexDatas
 */
/****************************************************************/
void Grid_Generation::InsertEdgeBndVertexData(const ZRGridLineDir theRayDir,
					      const Standard_Size theRayIndex,
					      const EdgeBndVertexData& aBndVertex)
{
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >* theVData = m_GridBndDatas->ModifyEdgeBndVertexDataOf(theRayDir);
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >::iterator it1 = theVData->find(theRayIndex);

  bool IsInserted = false;
  if(it1!=theVData->end()){
    vector<EdgeBndVertexData>& theVertices = it1->second;
    vector<EdgeBndVertexData>::iterator iter;
    for(iter=theVertices.begin(); iter!=theVertices.end(); iter++){
      if(iter->m_Index==aBndVertex.m_Index){
	if(iter->m_Frac==aBndVertex.m_Frac){
	  if(iter->TransitionType == 1){
	    theVertices.insert(iter, aBndVertex);
	    IsInserted = true;
	    break;
	  }else if(iter->TransitionType == -1){
	    vector<EdgeBndVertexData>::iterator nextIter = iter+1;
	    if(nextIter!=theVertices.end()){
	      theVertices.insert(nextIter, aBndVertex);
	    }else{
	      theVertices.push_back(aBndVertex);
	    }
	    IsInserted = true;
	    break;
	  }else{
	    cout<<"Grid_Generation::InsertEdgeBndVertexData--------Error--------Two EdgeBndVertexData's Index are same------Tangent"<<endl;
	    continue;
	  }
	}else if(iter->m_Frac>aBndVertex.m_Frac){
	  theVertices.insert(iter, aBndVertex);
	  IsInserted = true;
	  break;
	}else{
	  continue;
	}
      }else if(iter->m_Index>aBndVertex.m_Index){
	theVertices.insert(iter, aBndVertex);
	IsInserted = true;
	break;
      }else{
	continue;
      }
    }
    if(!IsInserted){
      theVertices.push_back(aBndVertex);
    }
  }else{
    vector<EdgeBndVertexData> v_PS;
    v_PS.push_back(aBndVertex);
    theVData->insert(pair<Standard_Size, vector<EdgeBndVertexData> >(theRayIndex, v_PS) );
  }
}
