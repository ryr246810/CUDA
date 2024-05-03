#include <GridVertexData.hxx>

#include <GridEdgeData.hxx>
#include <GridFaceData.cuh>
#include <GridEdge.hxx>
#include <GridFace.hxx>
#include <GridGeometry.hxx>
#include <T_Element.hxx>

void GridVertexData::ClearSharingDivTEdges()
{
  for(Standard_Integer dir = 0; dir<3; dir++){
    m_SharedDivTEdges[dir].clear();
  }
}



void GridVertexData::ClearSharingTDFaces()
{
  m_SharedTDFaces.clear();
}



void GridVertexData::ClearSharingGridFaceDatas()
{
  m_GridFaceDatas.clear();
}



bool Is_GridVertexData_As_End_Of_GridEdgeData(GridVertexData* inputVertex, GridEdgeData* inputEdge)
{
  bool result = false;
  if( (inputEdge->GetFirstVertex() == inputVertex) || (inputEdge->GetLastVertex() == inputVertex) ){
    result = true;
  }
  return result;
}



void GridVertexData::BuildDivTEdges()
{
  ClearSharingDivTEdges();
  //if(this->GetState()==BND) return; //tzh Modify 20210416

  Standard_Size theVertexVectorIndex[2];
  this->GetZRGrid()->FillVertexIndxVec(m_Index, theVertexVectorIndex);

  Standard_Integer Dir0, Dir1;

  for(Standard_Integer dir = 0; dir<2; dir++){
    Dir0 = dir;
    Dir1 = (Dir0 + 1)%2;
    
    Standard_Integer theLowerIndex_Dir0 = theVertexVectorIndex[Dir0]-1;
    Standard_Integer theUpperIndex_Dir0 = theVertexVectorIndex[Dir0];

    if(theVertexVectorIndex[Dir0] == GetZRGrid()->GetXtndRgn().getLowerBound(Dir0))  theLowerIndex_Dir0 = theVertexVectorIndex[Dir0];
    if(theVertexVectorIndex[Dir0] == GetZRGrid()->GetXtndRgn().getUpperBound(Dir0))  theUpperIndex_Dir0 = theVertexVectorIndex[Dir0]-1;

    for(Standard_Integer index = theLowerIndex_Dir0; index<=theUpperIndex_Dir0; index++){
      Standard_Size currEdgeVectorIndex[2];
      currEdgeVectorIndex[Dir0] = index;
      currEdgeVectorIndex[Dir1] = theVertexVectorIndex[Dir1];

      Standard_Size currEdgeSclarIndex;
      GetZRGrid()->FillEdgeIndx(Dir0, currEdgeVectorIndex, currEdgeSclarIndex);
      GridEdge* currGridEdge = m_GridGeom->GetGridEdges()[Dir0] + currEdgeSclarIndex;

      vector<GridEdgeData*> currGridEdgeDatas = currGridEdge->GetEdges();

      for(Standard_Size n=0; n<currGridEdgeDatas.size(); n++){
	//if(Is_GridVertexData_As_End_Of_GridEdgeData(this, currGridEdgeDatas[n])) //tzh Modify 20210416
	{
	  Standard_Integer relativeDir = 1;

	  if( currEdgeVectorIndex[Dir0]==theVertexVectorIndex[Dir0] ){
	    relativeDir = 1;
	  }else if( currEdgeVectorIndex[Dir0]==(theVertexVectorIndex[Dir0]-1) ){
	    relativeDir = -1;
	  }else{
	    continue;
	  }

	  T_Element tmpTEdge(currGridEdgeDatas[n], relativeDir);
	  m_SharedDivTEdges[Dir0].push_back(tmpTEdge);
	  break;
	}
      }
    }
  }
}


bool Is_GridVertexData_As_Vertex_Of_GridFaceData(GridVertexData* _vertex, GridFaceData* _face)
{
  bool result = false;

  vector<VertexData*> theAllVertexDatas;
  _face->GetOrderedVertexDatas(theAllVertexDatas);

  for(Standard_Size n=0; n<theAllVertexDatas.size(); n++){
    if(theAllVertexDatas[n]==_vertex){
      result = true;
      break;
    }
  }

  return result;
}


void GridVertexData::BuildSharingGridFaceDatas()
{
  ClearSharingGridFaceDatas();

  //if(this->GetState()==BND) return; // tzh Modify 20210416

  Standard_Size theVertexVectorIndex[3];
  this->GetZRGrid()->FillVertexIndxVec(m_Index, theVertexVectorIndex);

  Standard_Integer Dir0, Dir1, Dir2;

  //for(Standard_Integer dir = 0; dir<1; dir++){  // ---0
    //Dir0 = dir;

    // Dir1 = (Dir0 + 1)%3;
    // Dir2 = (Dir0 + 2)%3;

    Dir1 = 0;
    Dir2 = 1;

    Standard_Integer theLowerIndex[2];
    Standard_Integer theUpperIndex[2];

    theLowerIndex[Dir1] = theVertexVectorIndex[Dir1]-1;
    theUpperIndex[Dir1] = theVertexVectorIndex[Dir1];

    theLowerIndex[Dir2] = theVertexVectorIndex[Dir2]-1;
    theUpperIndex[Dir2] = theVertexVectorIndex[Dir2];

    if( theVertexVectorIndex[Dir1] ==  GetZRGrid()->GetXtndRgn().getLowerBound(Dir1) )   theLowerIndex[Dir1] = theVertexVectorIndex[Dir1];
    if( theVertexVectorIndex[Dir1] ==  GetZRGrid()->GetXtndRgn().getUpperBound(Dir1) )   theUpperIndex[Dir1] = theVertexVectorIndex[Dir1]-1;

    if( theVertexVectorIndex[Dir2] ==  GetZRGrid()->GetXtndRgn().getLowerBound(Dir2) )   theLowerIndex[Dir2] = theVertexVectorIndex[Dir2];
    if( theVertexVectorIndex[Dir2] ==  GetZRGrid()->GetXtndRgn().getUpperBound(Dir2) )   theUpperIndex[Dir2] = theVertexVectorIndex[Dir2]-1;

    for( Standard_Integer index1=theLowerIndex[Dir1]; index1<=theUpperIndex[Dir1]; index1++ ){
      for( Standard_Integer index2=theLowerIndex[Dir2]; index2<=theUpperIndex[Dir2]; index2++ ){
	
	Standard_Size currFaceVectorIndex[2];
	currFaceVectorIndex[Dir1] = index1;
	currFaceVectorIndex[Dir2] = index2;
	
	Standard_Size currFaceSclarIndex;
	GetZRGrid()->FillFaceIndx(currFaceVectorIndex, currFaceSclarIndex);
	
	GridFace* currGridFace = m_GridGeom->GetGridFaces()+ currFaceSclarIndex;
	const vector<GridFaceData*>& currGridFaceDatas = currGridFace->GetFaces();
	
	for(Standard_Integer m=0; m<currGridFaceDatas.size(); m++){
	  //if(Is_GridVertexData_As_Vertex_Of_GridFaceData(this, currGridFaceDatas[m])){ //tzh Modify 20210416 
	  {
	    m_GridFaceDatas.push_back(currGridFaceDatas[m]);
	    break;
	  }
	}
      }
    }
  //}
}




void GridVertexData::BuildSharedTDFaces()
{
  ClearSharingTDFaces();

  if(this->GetState()==BND) return;

  Standard_Size theVertexVectorIndex[2];
  this->GetZRGrid()->FillVertexIndxVec(m_Index, theVertexVectorIndex);

  Standard_Integer Dir0, Dir1;

  for(Standard_Integer dir = 0; dir<2; dir++){
    Dir0 = dir;
    Dir1 = (Dir0 + 1)%2;
    
    Standard_Integer theLowerIndex_Dir0 = theVertexVectorIndex[Dir0]-1;
    Standard_Integer theUpperIndex_Dir0 = theVertexVectorIndex[Dir0];

    if(theVertexVectorIndex[Dir0] == GetZRGrid()->GetXtndRgn().getLowerBound(Dir0))  theLowerIndex_Dir0 = theVertexVectorIndex[Dir0];
    if(theVertexVectorIndex[Dir0] == GetZRGrid()->GetXtndRgn().getUpperBound(Dir0))  theUpperIndex_Dir0 = theVertexVectorIndex[Dir0]-1;

    for(Standard_Integer index = theLowerIndex_Dir0; index<=theUpperIndex_Dir0; index++){
      Standard_Size currEdgeVectorIndex[2];
      currEdgeVectorIndex[Dir0] = index;
      currEdgeVectorIndex[Dir1] = theVertexVectorIndex[Dir1];

      Standard_Size currEdgeSclarIndex;
      GetZRGrid()->FillEdgeIndx(Dir0, currEdgeVectorIndex, currEdgeSclarIndex);
      GridEdge* currGridEdge = m_GridGeom->GetGridEdges()[Dir0] + currEdgeSclarIndex;

      vector<GridEdgeData*> currGridEdgeDatas = currGridEdge->GetEdges();

      for(Standard_Size n=0; n<currGridEdgeDatas.size(); n++){
	if(Is_GridVertexData_As_End_Of_GridEdgeData(this, currGridEdgeDatas[n])){
	  Standard_Integer relativeDir = 1;

	  if( currEdgeVectorIndex[Dir0]==theVertexVectorIndex[Dir0] ){
	    if(Dir0==0) relativeDir = 1;
	    else relativeDir = -1;
	  }else if( currEdgeVectorIndex[Dir0]==(theVertexVectorIndex[Dir0]-1) ){
	    if(Dir0==0) relativeDir = -1;
	    else relativeDir = 1;
	  }else{
	    continue;
	  }

	  T_Element tmpTEdge(currGridEdgeDatas[n], relativeDir);
	  m_SharedTDFaces.push_back(tmpTEdge);

	  currGridEdgeDatas[n]->AddDEdge(this, relativeDir);
	  break;
	}
      }
    }
  }

  /*
  if(this->IsMaterialType(PEC)){
    cout<<"m_SharedTDFaces.size() = "<<m_SharedTDFaces.size()<<endl;
  }
  //*/
}
