#include <EdgeData.hxx>

#include <AppendingVertexDataOfGridEdge.hxx>
#include <PhysConsts.hxx>

//#define LENGTH_DBG


EdgeData
::EdgeData() : DataBase()
{
  m_Length = 0.0;

  m_FirstVertex = NULL;
  m_LastVertex = NULL;
}


EdgeData
::EdgeData( Standard_Integer _mark) :DataBase(_mark)
{
  m_Length = 0.0;

  m_FirstVertex = NULL;
  m_LastVertex = NULL;
}


EdgeData::
~EdgeData()
{
}


void 
EdgeData::
SetVertices( VertexData* _firstVertex, VertexData* _lastVertex )
{
  m_FirstVertex = _firstVertex;
  m_LastVertex  = _lastVertex;
};


void 
EdgeData::
Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();
}


void 
EdgeData::
SetupGeomDimInf()
{
  ComputeLength();
  ComputeAreaOfSweptFace();
}


/*************************************************************************************/
/*************************************************************************************/

void 
EdgeData::
ComputeLength()
{
  TxVector2D<Standard_Real> tmpVector1 = m_FirstVertex->GetLocation();
  TxVector2D<Standard_Real> tmpVector2 = m_LastVertex->GetLocation();
  m_Length = (tmpVector2-tmpVector1).length();
}


void 
EdgeData::
ComputeAreaOfSweptFace()
{
  TxVector2D<Standard_Real> theMidPnt;
  this->ComputeMidPntLocation(theMidPnt);
  //cout<<this->GetLastVertex()->GetLocation()[1]-this->GetFirstVertex()->GetLocation()[1]<<endl;
  //getchar();
  Standard_Integer rDir = 1;
  Standard_Real R0 = fabs(theMidPnt[rDir]);

  TxVector2D<Standard_Real>  theVector = m_LastVertex->GetLocation() - m_FirstVertex->GetLocation();
  m_AreaOfSweptFace = 2.0*mksConsts.pi*R0*theVector.length();
}


void 
EdgeData::
ComputeNaturalVector(TxVector2D<Standard_Real> & result)
{
  TxVector2D<Standard_Real> tmpVector1 = m_FirstVertex->GetLocation();
  TxVector2D<Standard_Real> tmpVector2 = m_LastVertex->GetLocation();
  result = tmpVector2-tmpVector1;
}


void 
EdgeData::
ComputeReversalVector(TxVector2D<Standard_Real> & result)
{
  TxVector2D<Standard_Real> tmpVector1 = m_FirstVertex->GetLocation();
  TxVector2D<Standard_Real> tmpVector2 = m_LastVertex->GetLocation();
  result = tmpVector1-tmpVector2;
}


void 
EdgeData::
ComputeMidPntLocation(TxVector2D<Standard_Real>& theMidPnt)
{
  TxVector2D<Standard_Real> firstPnt = m_FirstVertex->GetLocation();
  TxVector2D<Standard_Real> lastPnt = m_LastVertex->GetLocation();
  theMidPnt = (firstPnt + lastPnt)/2.0;
}


VertexData* 
EdgeData::
GetFirstVertex(const Standard_Integer rdir)
{
  if(rdir==1){
    return m_FirstVertex;
  }else{
    return m_LastVertex;
  }
}


VertexData* 
EdgeData::
GetLastVertex(const Standard_Integer rdir)
{
  if(rdir==1){
    return m_LastVertex;
  }else{
    return m_FirstVertex;
  }
}


VertexData* 
EdgeData::
GetFirstVertex()
{ 
  return m_FirstVertex;  
}


VertexData* 
EdgeData::
GetLastVertex() 
{
  return m_LastVertex;  
}


Standard_Real 
EdgeData::
GetGeomDim() const 
{
  return m_Length;
}


Standard_Real 
EdgeData::
GetSweptGeomDim() const 
{
  return m_AreaOfSweptFace;
}


Standard_Real 
EdgeData::
GetDualGeomDim() const 
{
  return 0.0;
}


Standard_Real 
EdgeData::
GetDualSweptGeomDim() const
{
  return 0.0;
}

Standard_Real 
EdgeData::
GetSweptGeomDim_Near()
{
  return 0.0;
}
Standard_Real 
EdgeData::
GetDualGeomDim_Near()
{
  return 0.0;
}
