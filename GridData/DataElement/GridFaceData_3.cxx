#include <GridFaceData.cuh>
#include <GridEdgeData.hxx>
#include <T_Element.hxx>
#include <VertexData.hxx>

//#define FACEDATA_DEBUG


void GridFaceData::AddEdge(EdgeData* anEdge, Standard_Integer aDir)
{
  T_Element _TopoEdge(anEdge, aDir);
  m_EdgeElements.push_back( _TopoEdge );
}


VertexData* GridFaceData::GetFirstVertex() const
{
  Standard_Size n = 0;
  GridEdgeData* currEdgeData = (GridEdgeData*) m_EdgeElements[n].GetData();
  Standard_Integer currRelativeDir = m_EdgeElements[n].GetRelatedDir();

  VertexData* result = currEdgeData->GetFirstVertex(currRelativeDir);

  /*
  VertexData* result = NULL;
  if(currRelativeDir==1){
    result = currEdgeData->GetFirstVertex();
  }else{
    result = currEdgeData->GetLastVertex();
  }
  //*/

  return result;
}


VertexData* GridFaceData::GetLastVertex() const
{
  Standard_Size nb = m_EdgeElements.size();
  Standard_Size n = nb-1;
  GridEdgeData* currEdgeData = (GridEdgeData*) m_EdgeElements[n].GetData();
  Standard_Integer currRelativeDir = m_EdgeElements[n].GetRelatedDir();

  VertexData* result = currEdgeData->GetLastVertex(currRelativeDir);

  /*
  VertexData* result = NULL;
  if(currRelativeDir==1){
    result = currEdgeData->GetLastVertex();
  }else{
    result = currEdgeData->GetFirstVertex();
  }
  //*/

  return result;
}


EdgeData* GridFaceData::GetFirstEdge() const
{
  Standard_Size n = 0;
  GridEdgeData* currEdgeData = (GridEdgeData*) m_EdgeElements[n].GetData();
  return currEdgeData;
}


EdgeData* GridFaceData::GetLastEdge() const
{
  Standard_Size nb = m_EdgeElements.size();
  Standard_Size n = nb-1;
  GridEdgeData* currEdgeData = (GridEdgeData*) m_EdgeElements[n].GetData();
  return currEdgeData;
}


bool GridFaceData::IsContaining(EdgeData* _edge) const
{
  bool tmp = false;

  Standard_Size nb = m_EdgeElements.size();
  for(Standard_Size index=0; index<nb; index++){
    if(m_EdgeElements[index].GetData() == _edge){
      tmp = true;
    }
  }

  return tmp;
}


bool GridFaceData::IsContaining(VertexData* _vertex) const
{
  bool tmp = false;

  Standard_Size nb = m_EdgeElements.size();
  for(Standard_Size index=0; index<nb; index++){
    GridEdgeData* currEdgeData = (GridEdgeData*) m_EdgeElements[index].GetData();

    if( (currEdgeData->GetFirstVertex() ==  _vertex) || 
	(currEdgeData->GetLastVertex() == _vertex) ){
      tmp = true;
      break;
    }
  }

  return tmp;
}


const vector<T_Element>& GridFaceData::GetOutLineTEdge() const
{
  return m_EdgeElements; 
}

