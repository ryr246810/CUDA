#include <VertexData.hxx>
#include <PhysConsts.hxx>


VertexData::VertexData()
  :DataBase()
{
  ClearFaceIndices();
  ClearShapeIndices();
  ClearMatDataIndices();
}


VertexData::VertexData( Standard_Integer _mark )
  :DataBase(_mark)
{
  ClearFaceIndices();
  ClearShapeIndices();
  ClearMatDataIndices();
}


VertexData::~VertexData()
{
  ClearShapeIndices();
  ClearFaceIndices();
  ClearMatDataIndices();
}


bool 
VertexData::
IsRegular() const
{
  bool tmp = false;
  if(GetType() == REGVERTEX) tmp = true;
  return tmp;
}


bool 
VertexData::
IsAppendedToGridEdge() const
{
  bool tmp = false;
  if(GetType() == BNDVERTEXOFEDGE) tmp = true;
  return tmp;
}


bool 
VertexData::
IsAppendedToGridFace() const
{
  bool tmp = false;
  if(GetType() == BNDVERTEXOFFACE) tmp = true;
  return tmp;
}

void 
VertexData::
Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();
}

void 
VertexData::
SetupGeomDimInf()
{
  ComputeSweptGeomDim();
}


void 
VertexData::
ComputeSweptGeomDim()
{
  TxVector2D<Standard_Real> theLocation = this->GetLocation();

  Standard_Integer rDir = 1;
  Standard_Real R0 = fabs(theLocation[rDir]);
  m_LengthOfSweptEdge = 2.0*mksConsts.pi*R0;
}
