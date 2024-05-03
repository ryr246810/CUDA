
#ifndef _NodeFlds_Elec_RgnSetter_HeaderFile
#define _NodeFlds_Elec_RgnSetter_HeaderFile


// vpbase includes
#include <NodeFlds_IterHolder.hxx>
#include <RgnWalker.hxx>
#include <NodeFlds_LinSetter.hxx>
#include <NodeFldsBase.hxx>


class NodeFlds_Elec_RgnSetter : public NodeFlds_IterHolder
{
public:
  NodeFlds_Elec_RgnSetter(NodeFldsBase* vf);

  NodeFlds_Elec_RgnSetter(const NodeFlds_Elec_RgnSetter& vfs);

  virtual ~NodeFlds_Elec_RgnSetter(){}

  NodeFlds_Elec_RgnSetter& operator=(const NodeFlds_Elec_RgnSetter& vfs);

  void UpdateVertices();
  void UpdateVertex();

public:
  void SetEdgePhysDataIndex(int dynElecPhysDataIndex){
    m_DynElecPhysDataIndex = dynElecPhysDataIndex;
  }

private:
  int m_DynElecPhysDataIndex;
};

#endif

