#include <PhysConsts.hxx>
#include <PortDataFunc.hxx>

#include <ZRGrid_Ctrl.hxx>
#include <Grid_Tool.hxx>
#include <TxMakerMap.h>

/*
void ZRGrid_Ctrl::SetupGrid()
{
  Standard_Real theZOrg;
  vector<Standard_Real> theZLengths;
  SetupGrid(m_ZDir, theZOrg, theZLengths);

  Standard_Real theROrg;
  vector<Standard_Real> theRLengths;
  SetupGrid(m_RDir, theROrg, theRLengths);

  Standard_Real theRZOrg[2];
  theRZOrg[0] = theZOrg; 
  theRZOrg[1] = theROrg; 

  m_ZRGrid = new ZRGrid;
  m_ZRGrid->SetupGrid(theRZOrg, theZLengths, theRLengths, m_Margin, m_ExtendedNum, m_GeomResolutionRatio);
  m_ZRGrid->Setup();

  Standard_Real theXYZOrg[3];
  theXYZOrg[m_ZDir] = theZOrg;
  theXYZOrg[m_RDir] = theROrg;
  theXYZOrg[m_WorkPlaneDir] = m_Org[m_WorkPlaneDir];

  m_ZRDefine = new ZRDefine(theXYZOrg, m_ZDir, m_RDir);
}
//*/


void ZRGrid_Ctrl::SetupGrid()
{
  Standard_Real theZOrg;
  vector<Standard_Real> theZLengths;
  SetupGrid(m_ZDir, theZOrg, theZLengths);

  Standard_Real theROrg;
  vector<Standard_Real> theRLengths;
  SetupGrid(m_RDir, theROrg, theRLengths);

  Standard_Real theRZOrg[2];
  theRZOrg[0] = theZOrg; 
  theRZOrg[1] = theROrg; 

  m_ZRGrid = new ZRGrid;

  /*
  for(size_t i = 0; i < 2; ++i){
    theRZOrg[i] = theRZOrg[i] * m_UnitsSystem->GetRealUnitScaleOfLength();
  }
  for(size_t i = 0; i < theZLengths.size(); ++i){
    theZLengths[i] = theZLengths[i] * m_UnitsSystem->GetRealUnitScaleOfLength();
  }
  for(size_t i = 0; i < theRLengths.size(); ++i){
    theRLengths[i] = theRLengths[i] * m_UnitsSystem->GetRealUnitScaleOfLength();
  }
  //*/

  m_ZRGrid->SetupGrid(theRZOrg, theZLengths, theRLengths, m_Margin, m_ExtendedNum, m_GeomResolutionRatio);
  m_ZRGrid->Setup();

  Standard_Real theXYZOrg[3];
  theXYZOrg[m_ZDir] = theZOrg;
  theXYZOrg[m_RDir] = theROrg;
  theXYZOrg[m_WorkPlaneDir] = m_Org[m_WorkPlaneDir];
  /*
  for(size_t i = 0; i < 3; ++i){
    theXYZOrg[i] = theXYZOrg[i] * m_UnitsSystem->GetRealUnitScaleOfLength();
  }
  //*/
  m_ZRDefine = new ZRDefine(theXYZOrg, m_ZDir, m_RDir);
}



void 
ZRGrid_Ctrl::
SetupGrid(const Standard_Integer dir, Standard_Real& theOrg, vector<Standard_Real>& theNewLengths)
{
  Standard_Real theLDStep = m_GridTools[dir]->GetFirstStep();
  Standard_Real theRUStep = m_GridTools[dir]->GetLastStep();
  theOrg = m_GridTools[dir]->GetOrg();

  vector<Standard_Real> theLengths;
  theLengths.clear();
  const vector<Standard_Real>& theCoordVec = m_GridTools[dir]->GetResult();
  Standard_Size nb = theCoordVec.size();
  for(Standard_Size n=0; n<nb; n++){
    Standard_Real currLength = theCoordVec[n] - theOrg;
    theLengths.push_back(currLength);
  }

  Standard_Integer theLDExtendedStepNum = 0;
  Standard_Integer theRUExtendedStepNum = 0;
  ExtendRgnDefineAccordingPorts(dir, theLDExtendedStepNum, theRUExtendedStepNum);
  ExtendRgnDefineAccordingMargin(theLDExtendedStepNum, theRUExtendedStepNum);

  Standard_Real theLDExtendedLength = theLDStep*theLDExtendedStepNum;
  theOrg -= theLDExtendedLength;

  BuildNewLength(theLDExtendedStepNum, theLDStep,
		 theRUExtendedStepNum, theRUStep,
		 theLengths, theNewLengths);
}



// need to modify the xyz to rz
void
ZRGrid_Ctrl::
ExtendRgnDefineAccordingPorts(const Standard_Integer dir,
			      Standard_Integer& theLDExtendedStepNum, 
			      Standard_Integer& theRUExtendedStepNum)
{
  if(m_LowerBndsIsSetAsPort[dir]){
    theLDExtendedStepNum += GetExtendedNum();
    if(m_LowerBndsIsSetAsInputPort[dir]){
      theLDExtendedStepNum += Get_PortRgn_To_PMLRgn_Gap();
    }
  }

  if(m_UpperBndsIsSetAsPort[dir]){
    theRUExtendedStepNum += GetExtendedNum();
    if(m_UpperBndsIsSetAsInputPort[dir]){
      theRUExtendedStepNum += Get_PortRgn_To_PMLRgn_Gap();
    }
  }

}



void 
ZRGrid_Ctrl::
ExtendRgnDefineAccordingMargin(Standard_Integer& theLDExtendedStepNum, 
			       Standard_Integer& theRUExtendedStepNum)
{
  theLDExtendedStepNum += GetMargin();
  theRUExtendedStepNum += GetMargin();
}



void
ZRGrid_Ctrl::
BuildNewLength(const Standard_Integer theLDExtendedStepNum,
	       const Standard_Real theLDStep,
	       const Standard_Integer theRUExtendedStepNum,
	       const Standard_Real theRUStep,
	       const vector<Standard_Real>& theMidLengths,
	       vector<Standard_Real>& theNewLengths)
{
  theNewLengths.clear();

  for(Standard_Integer i=0; i<theLDExtendedStepNum; i++){
    Standard_Real tmpLength = ((Standard_Real)i)*theLDStep; 
    theNewLengths.push_back(tmpLength);
  }

  Standard_Real theLDExtendedLength = ((Standard_Real)theLDExtendedStepNum)*theLDStep;
  Standard_Integer nMid = (Standard_Integer)theMidLengths.size();
  for(Standard_Integer i=0; i<nMid; i++){
    Standard_Real tmpLength = theMidLengths[i] + theLDExtendedLength;
    theNewLengths.push_back(tmpLength);
  }

  Standard_Real theMidLength = theMidLengths[nMid-1];
  
  for(Standard_Integer i=1; i<=theRUExtendedStepNum; i++){
    Standard_Real tmpLength = ((Standard_Real)i) * theRUStep + theLDExtendedLength + theMidLength; 
    theNewLengths.push_back(tmpLength);
  }
}
