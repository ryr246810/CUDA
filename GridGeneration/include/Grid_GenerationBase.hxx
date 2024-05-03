#ifndef _Mesh_GenerationBase_HeaderFile
#define _Mesh_GenerationBase_HeaderFile

#include <ZRGrid.hxx>
#include <ZRDefine.hxx>

#include <TxSlab.h>

#include <map>

using namespace std;

class Grid_GenerationBase
{
public:
  Grid_GenerationBase();
  Grid_GenerationBase(ZRGrid* _zrg, ZRDefine* _zrd, const Standard_Integer _backgroundmaterialtype);
  ~Grid_GenerationBase();


  //build mesh
public:
  void SetBackGroundMaterialType(const Standard_Integer);

  Standard_Integer GetBackGroundMaterialType() const;
  const ZRGrid* GetZRGrid() const;
  const ZRDefine* GetZRDefine() const;


private:
  ZRGrid* m_ZRGrid;
  ZRDefine* m_ZRDefine;
  Standard_Integer m_BackGroundMaterialType;
};

#endif
