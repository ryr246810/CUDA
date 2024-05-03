#ifndef NodeFlds_Mag_HeaderFile
#define NodeFlds_Mag_HeaderFile

#include <NodeFlds_Mag_RgnSetter.hxx>

class NodeFlds_Mag: public NodeFldsBase
{
public:
  NodeFlds_Mag();

  NodeFlds_Mag(std::string nm, GridGeometry* gridGeom, size_t numComp);
  virtual ~NodeFlds_Mag();


public:
  virtual void SetPhysDataIndexInGridGeom(const Standard_Integer _index);
  virtual void SetupDataSetter();
  virtual void Update();//关键是这个update函数


protected:
  Standard_Integer m_DataIndexInGridGeom;
  NodeFlds_Mag_RgnSetter * m_DataSetter;
};




#endif