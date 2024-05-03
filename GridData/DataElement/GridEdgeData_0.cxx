
#include <GridEdge.hxx>


#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>



// From Surroundint GridEdgeDatas
void
GridEdgeData::
DeduceShapeIndices()
{
  m_ShapeIndices.clear();

  VertexData* firstVertex = GetFirstVertex();
  VertexData* lastVertex  = GetLastVertex();

  if(firstVertex->GetState() == BND){
    VertexData* firstNextVertex = NULL;
    if(m_MidVertices.empty()){
      firstNextVertex = lastVertex;
    }else{
      firstNextVertex = m_MidVertices[0];
    }
    AddShapeIndex(firstVertex, firstNextVertex);
  }else{
    AddShapeIndex(firstVertex);
  }


  Standard_Size nb = m_MidVertices.size();

  if(lastVertex->GetState() == BND){
    VertexData* lastPreVertex = NULL;
    if(m_MidVertices.empty()){
      lastPreVertex = firstVertex;
    }else{
      lastPreVertex = m_MidVertices[nb-1];
    }
    AddShapeIndex(lastVertex, lastPreVertex);
  }else{
    AddShapeIndex(lastVertex);
  }


  for(Standard_Size i=0; i<nb; i++){
    AddShapeIndex(m_MidVertices[i]);
  }


  /*
  if(m_ShapeIndices.empty()){
    cout<<"GridEdgeData::DeduceShapeIndices()-----------------------emptyShapeIndeices---------error inf----->>>"<<endl;
    cout<<"m_MidVertices.size()\t=\t"<<m_MidVertices.size()<<endl;
    cout<<"first state\t=\t"<<firstVertex->GetState()
	<<"\t type\t=\t"<<firstVertex->GetType()
	<<"\t\t shapeindices num \t=\t"<<firstVertex->GetShapeIndices().size()<<endl;
    cout<<"last state\t=\t"<<lastVertex->GetState()
	<<"\t type\t=\t"<<lastVertex->GetType()
	<<"\t\t shapeindices num \t=\t"<<lastVertex->GetShapeIndices().size()<<endl;
  }
  //*/
}



// From ShapeIndices
void
GridEdgeData::
DeduceMaterialType()
{
  Standard_Integer tmp_mt=0;

  if(m_ShapeIndices.empty()){
    tmp_mt = GetBaseGridEdge()->GetGridGeom()->GetBackGroundMaterialType();
  }else{
    for(set<Standard_Integer>::iterator iter=m_ShapeIndices.begin(); iter!=m_ShapeIndices.end(); iter++){
      Standard_Integer curr_ShapeIndex = *iter;
      Standard_Integer curr_mt =  GetBaseGridEdge()->GetGridGeom()->GetMaterialTypeWithShapeIndex( curr_ShapeIndex );
      tmp_mt = tmp_mt | curr_mt;
    }
  }

  this->SetMaterialType(tmp_mt);
}


void GridEdgeData::ResetShapeIndices()
{
  m_ShapeIndices.clear();
}


const set<Standard_Integer>& GridEdgeData::GetShapeIndices() const 
{ 
  return m_ShapeIndices;
};


void GridEdgeData::AddShapeIndex(Standard_Integer _index)
{
  m_ShapeIndices.insert(_index);
};


void GridEdgeData::AddShapeIndex(VertexData* firstV, VertexData* secondV)
{
  const set<Standard_Integer>& firstShapeIndices  = firstV->GetShapeIndices();
  const set<Standard_Integer>& secondShapeIndices = secondV->GetShapeIndices();

  set<Standard_Integer>::const_iterator iter;
  for(iter=firstShapeIndices.begin(); iter!=firstShapeIndices.end(); iter++){
    if(secondShapeIndices.find(*iter)!=secondShapeIndices.end()){
      AddShapeIndex(*iter);
    }
  }
}


void GridEdgeData::AddShapeIndex(VertexData* V)
{
  const set<Standard_Integer>& theShapeIndices  = V->GetShapeIndices();
  set<Standard_Integer>::const_iterator iter;
  for(iter=theShapeIndices.begin(); iter!=theShapeIndices.end(); iter++){
      AddShapeIndex(*iter);
  }
}


bool GridEdgeData::HasShapeIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_ShapeIndices.find(_index);
  if(iter!=m_ShapeIndices.end()) result = true;
  return result;
}
