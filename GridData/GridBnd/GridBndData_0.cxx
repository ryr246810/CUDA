#include <GridBndData.hxx>
#include <iostream>

void 
GridBndData::
SetEdgeBndVertices(const Standard_Integer theDir, 
		   const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >& theData)
{
  if(theDir==0){
    m_EdgeBndVertexData0.clear();
    m_EdgeBndVertexData0 = theData;
  }else if(theDir==1){
    m_EdgeBndVertexData1.clear();
    m_EdgeBndVertexData1 = theData;
  }else{
  }
}


void 
GridBndData::
SetFaceBndVertices(const vector<FaceBndVertexData>& theData)
{
  m_FaceBndVertexData = theData;
}


void 
GridBndData::
SetShapesType(const map<Standard_Integer, Standard_Integer>& theData)
{
  m_ShapesWithTypeTool.clear();
  m_ShapesWithTypeTool = theData;
}


void 
GridBndData::
SetFacesType(const map<Standard_Integer, Standard_Integer>& theData)
{
  m_FacesWithTypeTool.clear();
  m_FacesWithTypeTool = theData;
}

void 
GridBndData::
SetPorts(const map<Standard_Integer, PortData, less<Standard_Integer> >& theData)
{
  m_Ports.clear();
  m_Ports = theData;
}



void 
GridBndData::
SetShapesMask(const map<Standard_Integer, Standard_Integer>& theData)
{
  m_ShapeMaskWithIndexTool.clear();
  m_ShapeMaskWithIndexTool = theData;
}



void 
GridBndData::
SetFacesMask(const map<Standard_Integer, Standard_Integer>& theData)
{
  m_FaceMaskWithIndexTool.clear();
  m_FaceMaskWithIndexTool = theData;
}




void 
GridBndData::
SetRelationBetweenFaceAndShape(const map<Standard_Integer, Standard_Integer>& theData)
{
  m_FacesWithShapeTool.clear();
  m_FacesWithShapeTool = theData;
}


void 
GridBndData::
SetRelationBetweenEdgeAndFace(const map<Standard_Integer, vector<Standard_Integer> >& theData)
{
  m_EdgesWithFaceTool.clear();
  m_EdgesWithFaceTool = theData;



  /*
  cout<<"void GridBndData::SetRelationBetweenEdgeAndFace------------------------------------------>>>"<<endl;
  map<Standard_Integer, vector<Standard_Integer>, less<Standard_Integer> >::const_iterator tmpIter;
  for(tmpIter=theData.begin(); tmpIter!=theData.end(); tmpIter++){
    const vector<Standard_Integer>& theVecDatas = tmpIter->second;
    vector<Standard_Integer>::const_iterator iter;

    cout<<"tmpIter->first = "<<tmpIter->first<<" : ";
    for(iter=theVecDatas.begin(); iter!=theVecDatas.end(); iter++){

      cout<<"*iter = "<<*iter<< " ; ";
    }
    cout<<endl;
  }
  cout<<endl;
  //*/
}

void 
GridBndData::
SetRelationBetweenVertexAndEdge(const map<Standard_Integer, vector<Standard_Integer> >& theData)
{
  m_VerticesWithEdgeTool.clear();
  m_VerticesWithEdgeTool = theData;



  /*
  cout<<"void GridBndData::SetRelationBetweenVertexAndEdge------------------------------------------>>>"<<endl;
  map<Standard_Integer, vector<Standard_Integer>, less<Standard_Integer> >::const_iterator tmpIter;
  for(tmpIter=theData.begin(); tmpIter!=theData.end(); tmpIter++){
    const vector<Standard_Integer>& theVecDatas = tmpIter->second;
    vector<Standard_Integer>::const_iterator iter;

    cout<<"tmpIter->first = "<<tmpIter->first<<" : ";
    for(iter=theVecDatas.begin(); iter!=theVecDatas.end(); iter++){

      cout<<"*iter = "<<*iter<< " ; ";
    }
    cout<<endl;
  }
  //*/
}
