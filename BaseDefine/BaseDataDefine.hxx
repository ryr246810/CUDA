
#ifndef _BaseDataDefine_HeaderFile
#define _BaseDataDefine_HeaderFile

const double SPEED_OF_LIGHT = 299792458;
const double iSPEED_OF_LIGHT = 1. / SPEED_OF_LIGHT;
const double SPEED_OF_LIGHT_SQ = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
const double iSPEED_OF_LIGHT_SQ = iSPEED_OF_LIGHT * iSPEED_OF_LIGHT;

enum AllMark{
  ZERO_MASK        = 0x0000000,

  TYPE_ZERO_MASK   = 0xf000000,
  STATE_ONLY_MASK  = 0xf000000,

  TYPE_ONLY_MASK   = 0x0ffffff,
  STATE_ZERO_MASK  = 0x0ffffff,

  VERTEX_OLNY_MASK = 0x00000ff,
  VERTEX_ZERO_MASK = 0xfffff00,

  EDGE_ONLY_MASK   = 0x000ff00,
  EDGE_ZERO_MASK   = 0xfff00ff,

  FACE_ONLY_MASK   = 0x0ff0000,
  FACE_ZERO_MASK   = 0xf00ffff,

  EM_ONLY_MASK     = 0x000ffff,
  EM_ZERO_MASK     = 0xfff0000,

  PTCL_ONLY_MASK   = 0xfff0000,
  PTCL_ZERO_MASK   = 0x000ffff
};


enum ElementStateWithShape{
  OUTSHAPE         = 0x1000000,
  INSHAPE          = 0x2000000,
  BND              = 0x4000000
};


enum VertexType {
  REGVERTEX        = 0x0000001,
  BNDVERTEXOFEDGE  = 0x0000002,
  BNDVERTEXOFFACE  = 0x0000004,
  BNDVERTEXOFCELL  = 0x0000008,
};


enum EdgeType{
  ZEROEDGE         = 0x0000100,
  REGEDGE          = 0x0000200,
  PFEDGE           = 0x0000400,
};


enum FaceType{
  ZEROFACE         = 0x0010000,
  REGFACE          = 0x0020000,
  PFFACE           = 0x0040000,
  BNDFACEOFCELL    = 0x0080000
};


enum EMBndMark{
  EMFREESPACE      = 0x0000001,
  PEC              = 0x0000002,
  USERDEFINED      = 0x0000004,
  PML              = 0x0000008,
  OPENPORT         = 0x0000010,
  INPUTPORT        = 0x0000020,
  MUR              = 0x0000040,
  MURPORT          = 0x0000080,
  PECPORT          = 0x0000100,
  INPUTMURPORT     = 0x0000200
};



enum PtclBndMark {
  PTCLFREESPACE   = 0x0010000,
  FOIL            = 0x0020000,
  EMITTER0        = 0x0040000,
  EMITTER1        = 0x0080000,
  EMITTER2        = 0x0100000,
  EMITTER3        = 0x0200000,
  EMITTER4        = 0x0400000,
  EMITTER5        = 0x0800000
};

#endif
