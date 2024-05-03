#ifndef _Mesh_Write_TMP_HeaderFile
#define _Mesh_Write_TMP_HeaderFile


#include <Grid_Generation.hxx>

using namespace std;

class Mesh_Write
{
public:
  Mesh_Write();
  Mesh_Write( Grid_Generation* _Data);
  ~Mesh_Write();

public:
  void WriteIntPnts(const ZRGridLineDir aDir);
  void WriteFaceBndPnts();
  void WriteEdgeVertices();
  void WriteFaceVertices();
  void WriteGridLine();
private:
  Grid_Generation* m_Data;
};

#endif
