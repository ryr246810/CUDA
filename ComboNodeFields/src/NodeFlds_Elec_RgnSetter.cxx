#include <NodeFlds_Elec_RgnSetter.hxx>


NodeFlds_Elec_RgnSetter::
NodeFlds_Elec_RgnSetter(NodeFldsBase* cemf)
  : NodeFlds_IterHolder(cemf)
{
}


NodeFlds_Elec_RgnSetter::
NodeFlds_Elec_RgnSetter(const NodeFlds_Elec_RgnSetter& cemfs)
  : NodeFlds_IterHolder(cemfs)
{
}


NodeFlds_Elec_RgnSetter& 
NodeFlds_Elec_RgnSetter::
operator=(const NodeFlds_Elec_RgnSetter& cemfs)
{
  NodeFlds_IterHolder::operator=(cemfs);
  return *this;
}


void 
NodeFlds_Elec_RgnSetter::
UpdateVertices()
{
  ptrReset();
  RgnWalker<NodeFlds_Elec_RgnSetter >::walk_Vertex(this->m_rgn, this);

  /*
  ptrReset();
  size_t theDataSize = this->m_rsltIter.m_fieldConstPtr->m_DataSize;
  cout<<"theDataSize = "<<theDataSize<<endl;
  double* ptr_1 = this->m_rsltIter.m_indxPtr;
  for(size_t i=0; i<theDataSize; i++){
    size_t rem = i%3;
    cout<<ptr_1[i]<<"\t";
    if((i!=0) && (rem==0)){
      cout<<endl;
    }
  }
  //*/
  /*
  cout<<"----------------------------"<<endl;
  size_t theDataSize = this->m_rsltIter.m_fieldConstPtr->m_DataSize;
  double* ptr_2 = this->m_rsltIter.m_fieldConstPtr->m_Data;
  for(size_t i=0; i<theDataSize; i++){
    size_t rem = i%3;
    cout<<ptr_2[i]<<"\t";
    if((i!=0) && (rem==0)){
      cout<<endl;
    }
  }
  //*/
}


void 
NodeFlds_Elec_RgnSetter::
UpdateVertex()
{
  /*
  this->m_rgn.write(cout);
  cout<<"------------------------------------------------->>>"<<endl;
  //*/
  NodeFlds_LinSetter::edgeFieldLinSetter(this->m_rsltIter, m_DynElecPhysDataIndex);
}
