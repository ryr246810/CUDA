 
#include <NodeFlds_BufferWriter.hxx>


 /*** Update by walking over the region */
void 
NodeFlds_BufferWriter::
UpdateVertices()
{

  this->resetBuffer();
  m_fldIterPtr->ptrReset();

  //RgnWalker< NodeFlds_BufferWriter >::walk_XtndVertex(this->m_rgn, this);
  RgnWalker< NodeFlds_BufferWriter >::walk_Vertex(this->m_rgn, this);
}


void 
NodeFlds_BufferWriter::
UpdateCells()
{
  this->resetBuffer();
  this->m_fldIterPtr->ptrReset();
  RgnWalker<NodeFlds_BufferWriter>::walk_Cell(this->m_rgn, this);
}


void 
NodeFlds_BufferWriter::
UpdateVertex()
{
  size_t NDIM = 2;
  size_t n = this->m_fldIterPtr->getNumElements();

  for(size_t j=0; j<n; ++j) {
    *this->m_bufPtr = (*this->m_fldIterPtr)();
    //cout<<*this->m_bufPtr<<"\t";
    //cout<<(*this->m_fldIterPtr)()<<"\t";

    this->m_fldIterPtr->bump(NDIM);
    ++this->m_bufPtr;
  }
  //cout<<endl;
  this->m_fldIterPtr->iBump(NDIM, n);
}


void 
NodeFlds_BufferWriter::
UpdateCell()
{
  int NDIM = 2;
  int n = this->m_fldIterPtr->getNumElements();
  
  for(int j=0; j<n; ++j) {
    *this->m_bufPtr = (*this->m_fldIterPtr)();

    this->m_fldIterPtr->bump(NDIM);
    ++this->m_bufPtr;
  }
  this->m_fldIterPtr->iBump(NDIM, n);
}

