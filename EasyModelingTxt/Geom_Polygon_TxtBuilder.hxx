#ifndef _Geom_Polygon_TxtBuilder_HeaderFile
#define _Geom_Polygon_TxtBuilder_HeaderFile


#include <Geom_TxtBuilderBase.hxx>
#include <vector>


class Geom_Polygon_TxtBuilder: public Geom_TxtBuilderBase
{
public:
  Geom_Polygon_TxtBuilder();
  ~Geom_Polygon_TxtBuilder();
  virtual void InitVariable();
  virtual void SetAttrib(const TxHierAttribSet& tas);
  virtual void Build();

public:
  Standard_Integer           m_Type;
  Standard_Integer           m_PntNum;
  std::vector<Standard_Real> m_Pnts;
};

#endif
