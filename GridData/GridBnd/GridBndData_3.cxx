#include <GridBndData.hxx>


#include <BaseDataDefine.hxx>



bool 
GridBndData::
HasShapeMaterialDataIndex(const Standard_Integer theShapeIndex, Standard_Integer& theMatDataIndex) const
{
  bool result = false;
  map<Standard_Integer, Standard_Integer, less<Standard_Integer> >::const_iterator iter0 = m_ShapeWithMaterialDataIndexMap.find(theShapeIndex);

  if(iter0!=m_ShapeWithMaterialDataIndexMap.end()){
    theMatDataIndex = iter0->second;
    result = true;
  }

  return result;
}




bool 
GridBndData::
HasShapeMaterialData(const Standard_Integer theShapeIndex, ISOEMMatData& theData) const
{
  bool result = false;
  map<Standard_Integer, Standard_Integer, less<Standard_Integer> >::const_iterator iter0 = m_ShapeWithMaterialDataIndexMap.find(theShapeIndex);

  if(iter0!=m_ShapeWithMaterialDataIndexMap.end()){

    Standard_Integer theMatDataIndex = iter0->second;

    map<Standard_Integer, ISOEMMatData, less<Standard_Integer> >::const_iterator iter1 = m_MaterialDataIndexWithMaterialDataMap.find(theMatDataIndex);
    if(iter1!=m_MaterialDataIndexWithMaterialDataMap.end()){
      theData = iter1->second;
      result = true;
    }
  }

  return result;
}




bool 
GridBndData::
IsOnePtclBnd(const Standard_Integer theFaceIndex) const
{
  bool result = false;
  Standard_Integer theFaceMatDefine = this->GetMaterialTypeWithFaceIndex(theFaceIndex);
  if( ( (Standard_Integer)(theFaceMatDefine & EMITTER0) != 0 )  ){
    result = true;
  }
  return result;
}



const vector<Standard_Integer>& 
GridBndData::
GetMatDataIndicesOfSpaceDefine() const
{
  return m_SpaceMaterialData;
}



bool 
GridBndData::
HasMatDataWithMatIndex(const Standard_Integer theMatDataIndex, ISOEMMatData& theData) const
{
  bool result = false;
  map<Standard_Integer, ISOEMMatData, less<Standard_Integer> >::const_iterator iter1 = m_MaterialDataIndexWithMaterialDataMap.find(theMatDataIndex);
  if(iter1!=m_MaterialDataIndexWithMaterialDataMap.end()){
    theData = iter1->second;
    result = true;
  }else{

  }
  return result;
}

