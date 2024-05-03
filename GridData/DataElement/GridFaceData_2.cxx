#include <GridFaceData.cuh>
#include <GridFace.hxx>


#include <EdgeData.hxx>
#include <T_Element.hxx>
#include <VertexData.hxx>

#include <BaseFunctionDefine.hxx>

//#define FACEDATA_DEBUG

/*
void 
GridFaceData::
GetOrderedVertexDatas(vector<VertexData*>& theAllVertexDatas)const
{
  theAllVertexDatas.clear();

  const vector<T_Element>&  theOutLineEdges = this->GetOutLineTEdge();

  Standard_Integer nb = (Standard_Integer)theOutLineEdges.size();
  for(Standard_Integer i=0; i<nb; i++){  // 0 ==>>>
    Standard_Integer currIndex = i;
    Standard_Integer nextIndex = NextIndexOfCircularIndices(nb, currIndex);

    EdgeData* currEdge = (EdgeData*)theOutLineEdges[currIndex].GetData();
    EdgeData* nextEdge = (EdgeData*)theOutLineEdges[nextIndex].GetData();
    Standard_Integer currRelativeDir = theOutLineEdges[currIndex].GetRelatedDir();
    Standard_Integer nextRelativeDir = theOutLineEdges[nextIndex].GetRelatedDir();

    VertexData* currFirstV;
    if(currRelativeDir==1){
      currFirstV = currEdge->GetFirstVertex();
    }else{
      currFirstV = currEdge->GetLastVertex();
    }
    theAllVertexDatas.push_back(currFirstV);

    { // 1 ==>>>
      VertexData* currLastV;
      VertexData* nextFirstV;
      if(currRelativeDir==1){
	currLastV = currEdge->GetLastVertex();
      }else{
	currLastV = currEdge->GetFirstVertex();
      }

      if(nextRelativeDir==1){
	nextFirstV = nextEdge->GetFirstVertex();
      }else{
	nextFirstV = nextEdge->GetLastVertex();
      }

      if( currLastV != nextFirstV ){   // 2 ==>>>
	theAllVertexDatas.push_back(currLastV);  // can inlude the edge with it's index is nb-1
      } // 2 ==<<<
    } // 1 ==<<<
  } // 0 ==<<<
}
//*/

//*
void 
GridFaceData::
GetOrderedVertexDatas(vector<VertexData*>& theAllVertexDatas)const
{
  theAllVertexDatas.clear();

  const vector<T_Element>&  theOutLineEdges = this->GetOutLineTEdge();

  Standard_Integer nb = (Standard_Integer)theOutLineEdges.size();
  for(Standard_Integer i=0; i<nb; i++){  // 0 ==>>>
    Standard_Integer currIndex = i;
    Standard_Integer nextIndex = NextIndexOfCircularIndices(nb, currIndex);

    EdgeData* currEdge = (EdgeData*)theOutLineEdges[currIndex].GetData();
    EdgeData* nextEdge = (EdgeData*)theOutLineEdges[nextIndex].GetData();
    Standard_Integer currRelativeDir = theOutLineEdges[currIndex].GetRelatedDir();
    Standard_Integer nextRelativeDir = theOutLineEdges[nextIndex].GetRelatedDir();

    VertexData* currFirstV = currEdge->GetFirstVertex(currRelativeDir) ;
    theAllVertexDatas.push_back(currFirstV);

    VertexData* currLastV = currEdge->GetLastVertex(currRelativeDir);
    VertexData* nextFirstV = nextEdge->GetFirstVertex(nextRelativeDir);
    if( currLastV != nextFirstV ){
      theAllVertexDatas.push_back(currLastV);  // can inlude the edge with it's index is nb-1
    }
  } // 0 ==<<<
}
//*/


void 
GridFaceData::
ComputeArea()
{
  vector<VertexData*> theAllVertexDatas;
  GetOrderedVertexDatas(theAllVertexDatas);

  Standard_Size nb = theAllVertexDatas.size();

  TxVector2D<Standard_Real> p = TxVector2D<Standard_Real>(0.0,0.0);  // barycenter
  vector<TxVector2D<Standard_Real> > VertexVector;
  for(Standard_Size i=0; i<nb; i++){
    TxVector2D<Standard_Real> aPnt = theAllVertexDatas[i]->GetLocation();
    VertexVector.push_back( aPnt );
  }
  for(Standard_Size i=0; i<nb; i++){
    p += VertexVector[i]; 
  }
  p = p/Standard_Real(nb);


  TxVector2D<Standard_Real> p0,p1;
  TxVector2D<Standard_Real> tmpVector1, tmpVector2;

  Standard_Real tmpArea=0.0;
  Standard_Real theArea=0.0;

  for(Standard_Size index=0; index<nb; index++){
    Standard_Size firstIndex=index;
    Standard_Size lastIndex = NextIndexOfCircularIndices(nb, firstIndex);

    p0=theAllVertexDatas[firstIndex]->GetLocation();
    p1=theAllVertexDatas[lastIndex]->GetLocation();

    tmpVector1 = p - p0;
    tmpVector2 = p - p1;

    tmpArea = 0.5*fabs(tmpVector1[0]*tmpVector2[1]-tmpVector1[1]*tmpVector2[0]);
    theArea += tmpArea;
  }

  m_Area = theArea;
  //m_Area = this->GetBaseGridFace()->GetArea();//tzh modify 20210416

}


void GridFaceData::ComputeBaryCenter()
{
  vector<VertexData*> theAllVertexDatas;
  GetOrderedVertexDatas(theAllVertexDatas);

  Standard_Size nb = theAllVertexDatas.size();
  vector<TxVector2D<Standard_Real> > VertexVector;
  VertexVector.clear();
  for(Standard_Size i=0; i<nb; i++){
    TxVector2D<Standard_Real> aPnt = theAllVertexDatas[i]->GetLocation();
    VertexVector.push_back(aPnt);
  }
  
  TxVector2D<Standard_Real> theBaryCenter(0.0,0.0);
  for(Standard_Size i=0; i<nb; i++){
    theBaryCenter += VertexVector[i]; 
  }

  m_BaryCenter = theBaryCenter/Standard_Real(nb);

  //m_BaryCenter.write
  (cout);
}


const TxVector2D<Standard_Real>& GridFaceData::GetBaryCenter() const
{
  return m_BaryCenter;
}
