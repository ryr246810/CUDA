#include <GeomBndDataDefine.hxx>
#include <TxVector.h>



Standard_Real MAX3(Standard_Real a, 
		   Standard_Real b,
		   Standard_Real c)
{
  Standard_Real result=a;
  if(result<=b) result=b;
  if(result<=c) result=c;
  return result;
}


void ComputeProperBndOfShape(const TopoDS_Shape & aShape, 
			     Standard_Real& xmin, 
			     Standard_Real& ymin, 
			     Standard_Real& zmin, 
			     Standard_Real& xmax, 
			     Standard_Real& ymax, 
			     Standard_Real& zmax)
{
  cout<<"GeomBndDataDefine----------ComputeProperBndOfShape----------------------------->>>"<<endl;
  Bnd_Box tmpBox;
  BRepBndLib::Add(aShape, tmpBox);
  tmpBox.Get(xmin, ymin, zmin, xmax, ymax, zmax);

  //Standard_Real aDeflection = 0.005;
  Standard_Real aDeflection = 0.05;
  Standard_Real deflection;
  deflection= MAX3( xmax-xmin , ymax-ymin , zmax-zmin)*aDeflection;

  BRepMesh_IncrementalMesh Inc(aShape, aDeflection);
  Inc.Perform();


  TopoDS_Shape theTargetShape = Inc.Shape();
  Bnd_Box box;
  BRepBndLib::Add(theTargetShape, box);
  box.Get(xmin, ymin, zmin, xmax, ymax, zmax);

  cout<<"GeomBndDataDefine----------ComputeProperBndOfShape-----------------------------<<<"<<endl;
}




void SetFaceBndVertexFromFaceBndPntAndItsGridLocation(const FaceBndPntData& theFaceBndPnt,
						      const Standard_Size theGridFaceIndex, 
						      const Standard_Size theFrac1, 
						      const Standard_Size theFrac2,
						      FaceBndVertexData& theFaceBndVertex)
{
  theFaceBndVertex.m_Index = theGridFaceIndex;  // GridFace Scalar Index
  theFaceBndVertex.m_Frac1 = theFrac1;
  theFaceBndVertex.m_Frac2 = theFrac2;

  theFaceBndVertex.m_ShapeIndex = theFaceBndPnt.TheShapeIndex;
  theFaceBndVertex.m_EdgeIndex = theFaceBndPnt.TheEdgeIndex;

  theFaceBndVertex.MaterialType = theFaceBndPnt.MaterialType;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void SetEdgeBndVertexFromEdgeBndPntAndItsGridLocation(const EdgeBndPntData& theBndPnt,
						      const Standard_Size theGridEdgeIndex,
						      const Standard_Size theFrac,
						      EdgeBndVertexData& aBndVertex)
{
  aBndVertex.m_Index = theGridEdgeIndex;
  aBndVertex.m_Frac  = theFrac;

  aBndVertex.m_ShapeIndex = theBndPnt.TheShapeIndex;
  aBndVertex.m_FaceIndex = theBndPnt.TheFaceIndex;
	
  aBndVertex.TransitionType = IntegerTransitionType(theBndPnt.TransitionType);
  aBndVertex.MaterialType = theBndPnt.MaterialType;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void CopyEdgeBndVertexFrom(const EdgeBndVertexData& org,
			   EdgeBndVertexData& target)
{
  target.m_Index =  org.m_Index;
  target.m_Frac  = org.m_Frac;
  
  target.m_ShapeIndex = org.m_ShapeIndex;
  target.m_FaceIndex = org.m_FaceIndex;
  
  target.TransitionType = org.TransitionType;
  target.MaterialType = org.MaterialType;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void CopyFaceBndVertexFrom(const FaceBndVertexData& org,
			   FaceBndVertexData& target)
{
  target.m_Index = org.m_Index;
  target.m_Frac1 = org.m_Frac1;
  target.m_Frac2 = org.m_Frac2;

  target.m_ShapeIndex = org.m_ShapeIndex;
  target.m_EdgeIndex = org.m_EdgeIndex;

  target.MaterialType = org.MaterialType;
}


void SetupOnePortData(const Standard_Integer thePortIndex, 
		      const Standard_Integer thePortType, 
		      const ZRGridLineDir theLineDir,
		      const Standard_Integer theRelativeDir, 
		      const TxSlab2D<Standard_Size>& gridRgn,
		      PortData& tmpPort)
{
  tmpPort.m_Index = thePortIndex;
  tmpPort.m_Type = thePortType;
  tmpPort.m_Dir = Standard_Integer(theLineDir);
  tmpPort.m_RelativeDir = theRelativeDir;
  for(Standard_Integer index = 0; index<2; index++){
    tmpPort.m_LDCords[index] = gridRgn.getLowerBound(index);
    tmpPort.m_RUCords[index] = gridRgn.getUpperBound(index);
  }
}



Standard_Integer IntegerTransitionType(const IntCurveSurface_TransitionOnCurve TransitionType)
{
  Standard_Integer result = 0;
  switch (TransitionType)
    {
    case IntCurveSurface_Tangent:
      {
	result = 0;
	break;
      }
    case IntCurveSurface_In:
      {
	result = 1;
	break;
      }
    case IntCurveSurface_Out:
      {
	result = -1;
	break;
      }
    }
  return result;
}


IntCurveSurface_TransitionOnCurve OCCTransitionType(const Standard_Integer  TransitionType)
{
  IntCurveSurface_TransitionOnCurve result;

  if(TransitionType > 0 ){
    result = IntCurveSurface_In;
  }else if(TransitionType < 0){
    result = IntCurveSurface_Out;
  }else {
    result = IntCurveSurface_Tangent;
  }
  return result;
}
