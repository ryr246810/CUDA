#ifndef _NodeFldsConstBuffer_HeaderFile
#define _NodeFldsConstBuffer_HeaderFile


#include <NodeFlds_Iter.hxx>
#include <TxSlab.h>
#include <NodeFldsBase.hxx>

/*** holds the basic data needed to transfer data between a field and a buffer */

class NodeFlds_ConstBuffer 
{
 public:
  NodeFlds_ConstBuffer() {
    m_fldIterPtr = NULL;
    m_bufPtr = m_bufPtr0 = NULL;
  }

  virtual ~NodeFlds_ConstBuffer(){
    if(m_fldIterPtr!=NULL) delete m_fldIterPtr;
  }

  void setField(const NodeFldsBase* f){
    delete m_fldIterPtr;
    m_fldIterPtr = new NodeFlds_ConstIter(f);
  }

  void setRegion(const TxSlab2D<int>& r){
    m_rgn = r;
  }
  
  void setBuffer(Standard_Real* bp) {
    m_bufPtr = m_bufPtr0 = bp;
  }

  inline void resetBuffer() {
    m_bufPtr = m_bufPtr0;
  }

  inline void bump(size_t dir){
    m_fldIterPtr->bump(dir);
  }
  
  inline void iBump(size_t dir){
    m_fldIterPtr->iBump(dir);
  }
  
  inline void bump(size_t dir, int amt){
    m_fldIterPtr->bump(dir, amt);
  }
  
  inline void iBump(size_t dir, int amt){
    m_fldIterPtr->iBump(dir, amt);
  }

 protected:
  /** The region over which to transfer data */
  TxSlab2D<int> m_rgn;
  
  /** The pointer to the beginning of the data */
  Standard_Real* m_bufPtr0;
  
  /** The current pointer to the data */
  Standard_Real* m_bufPtr;

  /** Iterator pointing to the field to be read or added to */
  NodeFlds_ConstIter* m_fldIterPtr;


 private:
  // Private to prevent use
  NodeFlds_ConstBuffer(const NodeFlds_ConstBuffer&);
  NodeFlds_ConstBuffer& operator=(const NodeFlds_ConstBuffer&);

};

#endif
