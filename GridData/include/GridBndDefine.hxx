
#ifndef _GridBndDefine_HeaderFile
#define _GridBndDefine_HeaderFile


#include <Standard_TypeDefine.hxx>
#include <TxSlab.h>
#include <TxSlab2D.h>
#include <string>

using namespace std;



typedef struct EdgeBndVertexData{
  Standard_Integer m_ShapeIndex;
  Standard_Integer m_FaceIndex;
  Standard_Size m_Index;  // GridEdge Index
  Standard_Size m_Frac;
  Standard_Integer TransitionType;
  Standard_Integer MaterialType;
} EdgeBndVertexData;



// modified 2013.01.23
typedef struct FaceBndVertexData{
  Standard_Integer m_ShapeIndex;
  Standard_Integer m_EdgeIndex;
  Standard_Size m_Index;  // GridFace Index
  Standard_Size m_Frac1;
  Standard_Size m_Frac2;
  Standard_Integer MaterialType;
} FaceBndVertexData;


typedef struct PortData{
  Standard_Integer m_Type;
  Standard_Integer m_Index;

  Standard_Integer m_Dir;
  Standard_Integer m_RelativeDir;

  Standard_Size    m_LDCords[2];  // Left,  down
  Standard_Size    m_RUCords[2];  // right, up
} PortData;



typedef struct CompoundPortData{
  int m_PortIndex;
  PortData m_Data;
} CompoundPortData;



typedef struct CompoundEdgeBndVertexData{
  size_t m_IntLineIndex;
  EdgeBndVertexData m_Data;
} CompoundEdgeBndVertexData;



typedef struct CompoundFaceBndVertexData{
  size_t m_CutFaceIndex;
  FaceBndVertexData m_Data;
} CompoundFaceBndVertexData;



typedef struct PairIntInt{
  int m_FirstData;
  int m_SecondData;
} PairIntInt;


enum GridLineDir{DIRX =0, DIRY=1, DIRZ=2};
enum ZRGridLineDir{DIRRZZ =0, DIRRZR=1, DIRRZPHI=2};


string CompoundName(string basName, int index);
string CompoundName(string baseName, int index, string fileExt);

void ComputeAbsorbingRgnAccordingOpenPort(const PortData& thePort, TxSlab2D<Standard_Integer>& thePMLRgn);

void ComputePMLInfAccordingOpenPort(const PortData& thePort,
				    TxSlab2D<Standard_Integer>& theRgn,
				    Standard_Integer& theStartIndex,
				    Standard_Integer& theLayerNum);

Standard_Integer Get_PortRgn_To_PMLRgn_Gap();

void ComputePortStartIndex(const PortData& thePort,  Standard_Integer& theStartIndex);

void ComputeMurTypePortRgn(const PortData& thePort, TxSlab2D<Standard_Integer>& theMurPortRgn);

void ComputePortPhysStartIndex(const PortData& thePort, Standard_Integer& theInterfaceIndex);
#endif

