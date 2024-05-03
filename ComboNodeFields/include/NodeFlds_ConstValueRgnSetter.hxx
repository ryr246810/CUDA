
#ifndef _NodeFlds_ConstValueRgnSetter_HeaderFile
#define _NodeFlds_ConstValueRgnSetter_HeaderFile


// vpbase includes
#include <NodeFlds_IterHolder.hxx>
#include <RgnWalker.hxx>
#include <NodeFldsBase.hxx>


class NodeFlds_ConstValueRgnSetter : public NodeFlds_IterHolder
{
public:

  NodeFlds_ConstValueRgnSetter(NodeFldsBase* vf);

  NodeFlds_ConstValueRgnSetter(const NodeFlds_ConstValueRgnSetter& vfs);

  virtual ~NodeFlds_ConstValueRgnSetter(){}

  NodeFlds_ConstValueRgnSetter& operator=(const NodeFlds_ConstValueRgnSetter& vps);


  inline void SetInitialConstValue(Standard_Real _value){
    m_ConstValue = _value;
  };

  inline void UpdateVertices(){
    //ptrReset();
    RgnWalker<NodeFlds_ConstValueRgnSetter >::walk_Vertex(this->m_rgn, this);
  }
  

  inline void UpdateVertex(){
    this->m_rsltIter() = m_ConstValue;
  }


protected:
  Standard_Real m_ConstValue;
};

#endif
