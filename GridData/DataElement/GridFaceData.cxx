#include <GridFaceData.cuh>
#include <GridFace.hxx>
#include <GridGeometry.hxx>

#include <PhysConsts.hxx>


GridFaceData::
GridFaceData()
  :DataBase()
{
  m_BaseGFace = NULL;
  m_DualLength = 0.0;
}


GridFaceData::
GridFaceData(GridFace *_baseface)
  :DataBase()
{
  SetBaseGridFace(_baseface);
}


GridFaceData::
GridFaceData(Standard_Integer _mark)
  :DataBase(_mark)
{
  m_BaseGFace = NULL;
  m_DualLength = 0.0;
}


GridFaceData::
GridFaceData(GridFace *_baseface, Standard_Integer _mark)
  :DataBase(_mark)
{
  SetBaseGridFace(_baseface);
}


GridFaceData::
~GridFaceData()
{
  ClearAppendingEdge();
  m_EdgeElements.clear();
}


void
GridFaceData::
SetBaseGridFace(GridFace* _gridface)
{ 
  m_BaseGFace = _gridface;
  m_DualLength = m_BaseGFace->GetDualLength();
}


void 
GridFaceData::
Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();

  SetupAppendingEdge();

  /*
  {
    Standard_Size indxVec[2];
    this->GetBaseGridFace()->GetVecIndex(indxVec);
    cout<<"indxVec = [" <<indxVec[0]<<", "<<indxVec[1]<<"] ";
    cout<<"  faceAreaRatio = "<<this->FaceAreaRatio();
    cout<<"  area, dualLength = ["<< this->GetGeomDim()<<", "<<this->GetDualGeomDim()/2.0/mksConsts.pi<<" ] "<<endl;
  }
  //*/
}


void 
GridFaceData::
SetupGeomDimInf()
{
  ComputeArea();
  ComputeBaryCenter();

  DeduceState();
  DeduceType();
}


void 
GridFaceData::
SetupMaterialData()
{
  DataBase::SetupMaterialData();

  DeduceShapeIndices();
  DeduceMaterialType();

  DeduceMaterialData();
}


bool 
GridFaceData::
HasShapeIndex(Standard_Integer _index) const
{
  bool result = false;
  set<Standard_Integer>::iterator iter = m_ShapeIndices.find(_index);
  if(iter!=m_ShapeIndices.end()) result = true;
  return result;
}


const set<Standard_Integer>& 
GridFaceData::
GetShapeIndices() const 
{ 
  return m_ShapeIndices;
};



bool 
GridFaceData::
IsOutLineEdgePhysDataDefined() const
{
  bool result=true;
  Standard_Size nb = m_EdgeElements.size();
  for(Standard_Size index = 0; index<nb; index++){
    bool tmp = (m_EdgeElements[index].GetData())->IsPhysDataDefined();
    result = result && tmp;
  }
  return result;
}
