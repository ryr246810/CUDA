#include <GridFace.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>
#include <AppendingVertexDataOfGridFace.hxx>

#include <GridFaceData.cuh>
#include <GridGeometry.hxx>
#include <set>



void GridFace::CheckEdgeIndices(const vector<GridEdgeData*>& theAllEdges, 
				const vector<Standard_Size>& theUsedEdgeIndices,
				bool& isProper)
{
  isProper = true;

  if(theUsedEdgeIndices.empty()){
    isProper = false;
    //cout<<"GridFace::CheckEdgeIndices------------warning info--------empty"<<endl;
  }else if(theUsedEdgeIndices.size()==1){
    isProper = false;
    //cout<<"GridFace::CheckEdgeIndices------------warning info--------oneEdge"<<endl;
  }else if(theUsedEdgeIndices.size()==2){
    Standard_Size theFirstIndx = theUsedEdgeIndices[0];
    Standard_Size theSecondIndx = theUsedEdgeIndices[1];
    if( (theAllEdges[theFirstIndx]->GetBaseGridEdge())==(theAllEdges[theSecondIndx]->GetBaseGridEdge()) ){
      isProper = false;
      cout<<"GridFace::CheckEdgeIndices------------error info--------two coline Edges"<<endl;
    }else{
      isProper = true;
    }
  }else{
    isProper = true;
  }
}
