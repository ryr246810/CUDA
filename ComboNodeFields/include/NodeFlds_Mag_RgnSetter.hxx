#ifndef _NodeFlds_Mag_RgnSetter_HeaderFile
#define _NodeFlds_Mag_RgnSetter_HeaderFile

class NodeFlds_Mag_RgnSetter : public NodeFlds_IterHolder
{
public:
  NodeFlds_Mag_RgnSetter(NodeFldsBase* vf);

  NodeFlds_Mag_RgnSetter(const NodeFlds_Mag_RgnSetter& vfs);

  virtual ~NodeFlds_Mag_RgnSetter(){}

  NodeFlds_Mag_RgnSetter& operator=(const NodeFlds_Mag_RgnSetter& vfs);

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