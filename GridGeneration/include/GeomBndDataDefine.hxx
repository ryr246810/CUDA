
#ifndef _GridBndDataDefine_HeaderFile
#define _GridBndDataDefine_HeaderFile


#include <GridBndDefine.hxx>
#include <OCCInclude.hxx>
#include <Standard_TypeDefine.hxx>

#include <TxSlab.h>
#include <TxVector.h>

#include <TxSlab2D.h>
#include <TxVector2D.h>


#include <string>

using namespace std;

typedef struct EdgeBndPntData{
  Standard_Integer TheShapeIndex;
  Standard_Integer TheFaceIndex;
  gp_Pnt ThePnt;
  IntCurveSurface_TransitionOnCurve TransitionType;
  TopAbs_State StateType;
  Standard_Integer MaterialType;
} EdgeBndPntData;


typedef struct FaceBndPntData{
  Standard_Integer TheShapeIndex;
  Standard_Integer TheEdgeIndex;
  gp_Pnt ThePnt;
  Standard_Integer MaterialType;
} FaceBndPntData;



// 2012.03.16
typedef struct CellBndPntData{
  gp_Pnt ThePnt;
  Standard_Integer TheShapeIndex;
  Standard_Integer TheVertexIndex;
  Standard_Integer TheLocalIndexInCell;
  Standard_Integer MaterialType;
} CellBndPntData;



Standard_Real MAX3(Standard_Real a, Standard_Real b,  Standard_Real c);

void ComputeProperBndOfShape(const TopoDS_Shape & aShape, 
			     Standard_Real& xmin, Standard_Real& ymin, Standard_Real& zmin,
			     Standard_Real& xmax, Standard_Real& ymax, Standard_Real& zmax);

void SetEdgeBndVertexFromEdgeBndPntAndItsGridLocation(const EdgeBndPntData& theBndPnt,
						      const Standard_Size theGridEdgeIndex, 
						      const Standard_Size theFrac,
						      EdgeBndVertexData& aBndVertex);

void SetFaceBndVertexFromFaceBndPntAndItsGridLocation(const FaceBndPntData& theFaceBndPnt,
						      const Standard_Size theGridFaceIndex,
						      const Standard_Size theFrac1,
						      const Standard_Size theFrac2,
						      FaceBndVertexData& aFaceBndVertex);



void SetupOnePortData(const Standard_Integer thePortIndex, 
		      const Standard_Integer thePortType, 
		      const ZRGridLineDir theLineDir,
		      const Standard_Integer theRelativeDir, 
		      const TxSlab2D<Standard_Size>& gridRgn,
		      PortData& tmpPort);


void CopyEdgeBndVertexFrom(const EdgeBndVertexData& org,
			   EdgeBndVertexData& target);

void CopyFaceBndVertexFrom(const FaceBndVertexData& org,
			   FaceBndVertexData& target);


Standard_Integer IntegerTransitionType(const IntCurveSurface_TransitionOnCurve TransitionType);
IntCurveSurface_TransitionOnCurve OCCTransitionType(const Standard_Integer  TransitionType);

#endif
