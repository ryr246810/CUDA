#ifndef _NodeFlds_Elec_HeaderFile
#define _NodeFlds_Elec_HeaderFile

#include <NodeFldsBase.hxx>
#include <NodeFlds_Elec_RgnSetter.hxx>

class NodeFlds_Elec: public NodeFldsBase
{
public:
  NodeFlds_Elec();

  NodeFlds_Elec(std::string nm, GridGeometry* gridGeom, size_t numComp);
  virtual ~NodeFlds_Elec();


public:
  virtual void SetPhysDataIndexInGridGeom(const Standard_Integer _index);
  virtual void SetupDataSetter();
  virtual void Update();


protected:
  Standard_Integer m_DataIndexInGridGeom;
  NodeFlds_Elec_RgnSetter * m_DataSetter;
};

#endif

