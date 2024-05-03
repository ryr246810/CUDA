
#ifndef _GridFaceData_Headerfile
#define _GridFaceData_Headerfile

#include <TxVector2D.h>
#include <T_Element.hxx>

#include <GridBndDefine.hxx>
#include <DataBase.hxx>

#include "../../Cuda_Files/CUDAHeader.cuh"

#include <set>
using namespace std;


class GridFace;
class GridEdge;


class VertexData;
class GridVertexData;
class AppendingVertexDataOfGridEdge;
class AppendingVertexDataOfGridFace;

class EdgeData;
class AppendingEdgeData;


//class T_Element;

class GridFaceData: public DataBase
{
public:
   GridFaceData();
   GridFaceData(Standard_Integer _mark);
   GridFaceData(GridFace* _baseface);
   GridFaceData(GridFace* _baseface, Standard_Integer _mark);
  ~GridFaceData();

public:
   virtual void Setup();
   virtual void SetupGeomDimInf();
   virtual void SetupMaterialData();


public:
   void ComputeArea();
   void ComputeBaryCenter();


private:
   void DeduceShapeIndices();
   void DeduceMaterialType();
   void DeduceState();
   void DeduceType();
   void DeduceMaterialData();


public:
   bool HasShapeIndex(Standard_Integer _index) const;
   const set<Standard_Integer>&  GetShapeIndices() const;


public:
   void       SetBaseGridFace(GridFace* _gridface);
   GridFace*  GetBaseGridFace() const{ return m_BaseGFace; };
   void SetLocalIndex(const Standard_Integer theIndex){m_LocalIndex = theIndex;};
   Standard_Integer GetLocalIndex() const {return m_LocalIndex;};

   void SetupAppendingEdge();


private:
   void SetupAppendingEdge_Tool(VertexData* _firstV, VertexData* _lastV, 
			       vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices);

   void SetupOneAppendingEdge(VertexData* _firstV, VertexData* _secondV);

   void GetBndVerticesOfGridFace(VertexData* firstV,
				VertexData* lastV,
				vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices,
				bool& founded);

   void GetBndVerticesOfGridFace_Tool(VertexData* theRefVertex,
				     vector<AppendingVertexDataOfGridFace*>& theInputFaceBndVertices,
				     vector<AppendingVertexDataOfGridFace*>& theFaceBndVertices);
   void ClearAppendingEdge();


public:
   const vector<AppendingEdgeData*>& GetAppendingEdgeDatas() const;

public:
   virtual Standard_Real GetGeomDim() const {
    return m_Area;
  }

   virtual Standard_Real GetGeomDimInv() const {
    return m_AreaInv;
  }

   virtual Standard_Real GetDualGeomDim() const {
    return m_DualLength;
  }

   virtual Standard_Real GetSweptGeomDim() const {
    return m_SweptVolume;
  }

   virtual Standard_Real GetDualSweptGeomDim() const{
    return 0.0;
  }


   Standard_Real FaceAreaRatio() const;
   bool IsSmallArea() const;



public:
   void AddEdge(EdgeData*, Standard_Integer);
   const vector<T_Element>& GetOutLineTEdge() const;

   bool IsOutLineEdgePhysDataDefined() const;


public:
   void GetOrderedVertexDatas(vector<VertexData*>& theAllVertexDatas)const;


   const TxVector2D<Standard_Real>& GetBaryCenter() const;


public:
   bool IsContaining(EdgeData* _edge) const;
   bool IsContaining(VertexData* _vertex) const;

   VertexData* GetFirstVertex() const;
   VertexData* GetLastVertex() const;

   EdgeData*   GetFirstEdge() const;
   EdgeData*   GetLastEdge() const;



private:
  GridFace* m_BaseGFace;
  Standard_Integer m_LocalIndex;

  set<Standard_Integer> m_ShapeIndices;

  Standard_Real m_DualLength;
  Standard_Real m_Area;
  Standard_Real m_AreaInv;
  Standard_Real m_SweptVolume;

  vector<AppendingEdgeData*> m_AppendingEdges;
  vector<T_Element> m_EdgeElements;

  TxVector2D<Standard_Real> m_BaryCenter;
};

#endif

