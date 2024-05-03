#ifndef _EdgeData_Headerfile
#define _EdgeData_Headerfile

#include <VertexData.hxx>
#include <DataBase.hxx>

class EdgeData: public DataBase
{
public:
  EdgeData();
  EdgeData(Standard_Integer _mark);
  ~EdgeData();

  virtual void SetVertices( VertexData* _firstVertex, VertexData* _lastVertex );
 

public:
  virtual void Setup();
  virtual void SetupGeomDimInf();

  virtual void ComputeAreaOfSweptFace();
  virtual void ComputeLength();


  /*********************Compute & Get Geometry Location Data*********/
public:
  void ComputeMidPntLocation(TxVector2D<Standard_Real>& theMidPnt);

  void ComputeNaturalVector(TxVector2D<Standard_Real> & result);
  void ComputeReversalVector(TxVector2D<Standard_Real> & result);

  VertexData* GetFirstVertex();
  VertexData* GetLastVertex();

  VertexData* GetFirstVertex(const Standard_Integer rdir);
  VertexData* GetLastVertex(const Standard_Integer rdir);


public:
  virtual Standard_Real GetGeomDim() const;
  virtual Standard_Real GetSweptGeomDim() const;
  virtual Standard_Real GetDualGeomDim() const;
  virtual Standard_Real GetDualSweptGeomDim() const;
  virtual Standard_Real GetSweptGeomDim_Near() ;
  virtual Standard_Real GetDualGeomDim_Near() ;


protected:
  Standard_Real m_Length;              // edge length
  Standard_Real m_AreaOfSweptFace;


protected:
  VertexData* m_FirstVertex;
  VertexData* m_LastVertex;
};

#endif
