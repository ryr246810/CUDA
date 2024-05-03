// --------------------------------------------------------------------
// File:	NodeFlds_LinSetter.cxx
// Purpose:	Go recursively through the dimensions and linearly interpolate two values of the field
// --------------------------------------------------------------------

#include <NodeFlds_LinSetter.hxx>


void 
NodeFlds_LinSetter::
edgeFieldLinSetter(NodeFlds_Iter& iter, const int dynElecPhysDataIndex)
{
  size_t NDIM = 2;
  
  size_t iVec[2];
  iter.fillXtndIndexVec(iVec);

  //cout<<"iVec0 = ["<<iVec[0]<<", "<<iVec[1]<<"]\t";

  for(size_t dir=0; dir<NDIM; dir++){  // Ez, Er
    size_t indx;
    iter.GetZRGrid()->FillEdgeIndx(dir, iVec, indx);
    GridEdge* currGridEdgePtr = (iter.GetGridGeom())->GetGridEdges()[dir] + indx;
    
    Standard_Real physValue = 0.0;
    
    const vector<GridEdgeData*> & currEdges = currGridEdgePtr->GetEdges();
    
    Standard_Size nb = currEdges.size();
    for(Standard_Size i=0; i<nb; i++){
      physValue += currEdges[i]->GetPhysData(dynElecPhysDataIndex)/((Standard_Real)nb);
    }
    iter() = physValue;
    iter.bump(NDIM);
  }

  { // Ephi
    size_t indx;
    iter.GetZRGrid()->FillVertexIndx(iVec, indx);
    GridVertexData* currGridVertexPtr = (iter.GetGridGeom())->GetGridVertices() + indx;
    Standard_Real physValue = currGridVertexPtr->GetSweptPhysData(dynElecPhysDataIndex);
    iter() = physValue;
    iter.bump(NDIM);
  }

  //iter.fillXtndIndexVec(iVec);
  //cout<<"iVec1 = ["<<iVec[0]<<", "<<iVec[1]<<"]\t";

  iter.iBump(NDIM, 3);

  //iter.fillXtndIndexVec(iVec);
  //cout<<"iVec2 = ["<<iVec[0]<<", "<<iVec[1]<<"]\t"<<endl;
}
