#include <GridBndData.hxx>
#include <TxStreams.h>

void 
GridBndData::
ConvertFaceMaskVectoIndexVec(const vector<Standard_Integer>& maskVec, vector<Standard_Integer>& indexVec) const
{
  indexVec.clear();
  if(maskVec.size()>0){
    const map<Standard_Integer, Standard_Integer>*  theFacesMaskIndexMap = this->GetFacesMask();
    map<Standard_Integer, Standard_Integer>::const_iterator iter;
    
    for(Standard_Size i=0; i<maskVec.size(); i++){
      Standard_Integer tmpMask = maskVec[i];
      iter = theFacesMaskIndexMap->find(tmpMask);
      if(iter != theFacesMaskIndexMap->end()){
	Standard_Integer tmpIndex = iter->second;
	indexVec.push_back(tmpIndex);
      }
    }
  }
}


void 
GridBndData::
ConvertFaceMasktoIndex(const Standard_Integer theMask, Standard_Integer& theIndex) const
{
  const map<Standard_Integer, Standard_Integer>*  theFacesMaskIndexMap = this->GetFacesMask();
  map<Standard_Integer, Standard_Integer>::const_iterator iter;
  
  iter = theFacesMaskIndexMap->find(theMask);
  if(iter != theFacesMaskIndexMap->end()){
    theIndex = iter->second;
  }else{
    theIndex = 0;
  }
}



void 
GridBndData::
ConvertShapeMasktoIndex(const Standard_Integer theMask, Standard_Integer& theIndex) const
{
  const map<Standard_Integer, Standard_Integer>*  theShapesMaskIndexMap = this->GetShapesMask();
  map<Standard_Integer, Standard_Integer>::const_iterator iter;
  
  iter = theShapesMaskIndexMap->find(theMask);
  if(iter != theShapesMaskIndexMap->end()){
    theIndex = iter->second;
  }else{
    theIndex = 0;
  }
}






void 
GridBndData::
ConvertFaceIndextoMask(const Standard_Integer theIndex, Standard_Integer& theMask) const
{
  const map<Standard_Integer, Standard_Integer>*  theFacesMaskIndexMap = this->GetFacesMask();
  map<Standard_Integer, Standard_Integer>::const_iterator iter;
  
  theMask = 0;

  for(iter = theFacesMaskIndexMap->begin(); iter!=theFacesMaskIndexMap->end(); iter++){
    if(theIndex == iter->second){
      theMask = iter->first;
      break;
    }
  }
}

