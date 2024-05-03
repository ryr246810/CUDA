
#include <PhysConsts.hxx>
#include <PortDataFunc.hxx>

#include <ZRGrid_Ctrl.hxx>
#include <Grid_Tool.hxx>
#include <TxMakerMap.h>



void 
ZRGrid_Ctrl::
Build()
{
  map<Standard_Integer, Grid_Tool*, less<Standard_Integer> >::iterator iter;
  for(iter=m_GridTools.begin(); iter!=m_GridTools.end(); iter++){
    Grid_Tool* currPtr = iter->second;
    currPtr->Build();
  }
  CheckAndModifyDiscreteCoords();//tzh Modify 20210416
}


void ZRGrid_Ctrl::CheckAndModifyDiscreteCoords()
{
  CheckAndModifyDiscreteCoords(m_ZDir);
}


void ZRGrid_Ctrl::CheckAndModifyDiscreteCoords(const Standard_Integer dir)
{
  const map<Standard_Real, Standard_Integer, less<Standard_Real> >* theSpecialCorrdsPtr = m_ModelsCtrl->GetSpecialCoordDatas(dir);
  const vector<Standard_Real>* theSpecialCorrdsVecPtr = m_ModelsCtrl->GetSpecialCoordVec(dir);
  
  vector<Standard_Real>& theCoordVec = m_GridTools[dir]->ModifyResult();
  vector<Standard_Real>::iterator iter;
  vector<Standard_Real>::iterator next_iter;
  vector<Standard_Real>::iterator pre_iter;
  for(iter=theCoordVec.begin(); iter!=theCoordVec.end(); iter++){
    Standard_Real& currCoord = *iter;
    Standard_Real currStep = 0.0;
    next_iter = iter+1;
    
    if(next_iter!=theCoordVec.end()){
      currStep = *next_iter - *iter;
    }else{
      pre_iter = iter-1;
      currStep = *iter - *pre_iter;
    }
    
    map<Standard_Real, Standard_Integer, less<Standard_Real> >::const_iterator mapIter;
    mapIter = theSpecialCorrdsPtr->upper_bound(currCoord);
    
    if(mapIter!=theSpecialCorrdsPtr->end()) {
      const Standard_Integer& refIndex = mapIter->second;
      if(refIndex>0){
	Standard_Integer firstIndex = refIndex-1;
	Standard_Integer lastIndex = refIndex;
	Standard_Real firstCoord = (*theSpecialCorrdsVecPtr)[firstIndex];
	Standard_Real lastCoord = (*theSpecialCorrdsVecPtr)[lastIndex];
	
	CheckAndModifyDiscreteCoord(firstCoord, currStep, currCoord);
	CheckAndModifyDiscreteCoord(lastCoord,  currStep, currCoord);
      }else{
	const Standard_Integer& refIndex = mapIter->second;
	Standard_Real refCoord = (*theSpecialCorrdsVecPtr)[refIndex];
	
	CheckAndModifyDiscreteCoord(refCoord, currStep, currCoord);
      }
    }
  }
}


void 
ZRGrid_Ctrl::
CheckAndModifyDiscreteCoord(const Standard_Real refCoord, 
			    const Standard_Real currStep, 
			    Standard_Real& currCoord)
{
  Standard_Real theRefTol = 0.01*currStep/m_GeomResolutionRatio;
  Standard_Real theDiff = fabs(refCoord-currCoord);
  
  if(theDiff<theRefTol){
    cout<<"\t\t currCoord modified from   "<< currCoord;
    currCoord = currCoord - 10.0*theRefTol;
    cout<<"   to   "<<currCoord<<endl;
  }
}
