
#include <GridEdge.hxx>
#include <T_Element.hxx>

#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>

#include <PhysConsts.hxx>



void 
GridEdgeData::
InitEfficientLength()
{
  m_EfficientLength = EdgeData::GetGeomDim();
}


/*
// this function can be called after all GridFaceDatas are constructed
//*/
void 
GridEdgeData::
ComputeEfficientLength()
{
  Standard_Real minLengthRatio = EdgeLengthRatio(); 

  const vector<T_Element>&  theSharedFaces = GetSharedTFace();
  vector<T_Element>::const_iterator iter;
  for(iter=theSharedFaces.begin(); iter!=theSharedFaces.end(); iter++){

    GridFaceData* currFaceData = (GridFaceData*)(iter->GetData());
    Standard_Real tmpRatio = 2.0*currFaceData->FaceAreaRatio();
    if( tmpRatio<minLengthRatio ){
      minLengthRatio = tmpRatio;
    }else{
      continue;
    }
  }

  m_EfficientLength = minLengthRatio * (GetBaseGridEdge()->GetLength());
}


void 
GridEdgeData::
ComputeDualLengthOfSweptFace()
{
  Standard_Integer dir = GetDir();
  Standard_Size indxVec[2];
  Standard_Size indx = GetBaseGridEdge()->GetIndex();
  m_BaseGEdge->GetZRGrid()->FillEdgeIndxVec(dir, indx, indxVec);

  Standard_Integer dualDir = (dir+1)%2;
  m_DualLengthOfSweptFace = m_BaseGEdge->GetZRGrid()->GetDualStep(dualDir, indxVec[dualDir]);
}


Standard_Real GridEdgeData::EdgeLengthRatio()
{
  Standard_Real theRealLength = EdgeData::GetGeomDim();
  Standard_Real tmpRatio = theRealLength/this->GetBaseGridEdge()->GetLength();
  return tmpRatio;
}


bool GridEdgeData::IsNotPartial()
{
  bool tmp = false;
  if( GetType() == REGEDGE){
    tmp = true;
  }
  return tmp;
}


bool GridEdgeData::IsPartial()
{
  bool tmp = false;
  if( GetType() == PFEDGE){
    tmp = true;
  }
  return tmp;
}
