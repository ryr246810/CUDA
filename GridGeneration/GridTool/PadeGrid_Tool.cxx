#include <PadeGrid_Tool.hxx>


PadeGrid_Tool::
PadeGrid_Tool()
  : Grid_Tool()
{

}


PadeGrid_Tool::
~PadeGrid_Tool()
{
  ClearTools();
}


void 
PadeGrid_Tool::
SetAttrib(const TxHierAttribSet& tas)
{
  Grid_Tool::SetAttrib(tas);

  std::vector<Standard_Real> theCoords;
  std::vector<Standard_Integer> theSteps;

  if(tas.hasPrmVec("coordVec")){
    theCoords = tas.getPrmVec("coordVec");
  }
  if(tas.hasOptVec("waveResolutionRatioVec")){
    theSteps = tas.getOptVec("waveResolutionRatioVec");
  }

  Standard_Size np = theCoords.size();
  Standard_Size ndL = theSteps.size();

  if(np!=ndL){
    cout<<"error--------PadeGrid_Tool::SetAttrib--------";
    cout<<"number of coordVec is not equal to one of waveResolutionRatioVec"<<endl;
    exit(1);
  }

  m_PadePntVec.clear();
  for(Standard_Size i=0; i<np; i++){
    PadePnt tmpPnt;
    tmpPnt.m_Coord = theCoords[i];
    tmpPnt.m_Step = theSteps[i];
    m_PadePntVec.push_back(tmpPnt);
  }

  SetupGridGenerationTool();
}


void 
PadeGrid_Tool::
SetupGridGenerationTool()
{
  Standard_Real theFirstCoord;
  Standard_Real theLength;
  Standard_Integer theFirstResolution;
  Standard_Integer theLastResolution;

  const TxSlab<Standard_Real>& theBndBox = m_GridCtrl->GetBndBox();
  Standard_Real theWaveLength = m_GridCtrl->GetMinWaveLength();

  vector<PadePnt>::iterator iter;
  vector<PadePnt>::iterator next_iter;

  for(iter=m_PadePntVec.begin(); iter!=m_PadePntVec.end(); iter++){
    PadePnt currPnt = *iter;

    if(iter==m_PadePntVec.begin()){
      Standard_Integer theResolution = currPnt.m_Step;
      theFirstResolution = theResolution;
      theLastResolution = theResolution;

      theFirstCoord  =  theBndBox.getLowerBound(m_Dir) - 
	0.25*theWaveLength/Standard_Real(theResolution);

      theLength = currPnt.m_Coord - theFirstCoord;

      PadeGridGenerationTool *oneNewTool = new PadeGridGenerationTool(theFirstCoord,
								      theLength, 
								      theFirstResolution, 
								      theLastResolution,
								      theWaveLength);

      m_ToolVec.push_back(oneNewTool);
    }

    next_iter = iter+1;
    if(next_iter!=m_PadePntVec.end()){
      PadePnt nextPnt = *next_iter;
      theFirstResolution = currPnt.m_Step;
      theLastResolution = nextPnt.m_Step;

      theFirstCoord = currPnt.m_Coord;
      theLength = nextPnt.m_Coord-theFirstCoord;

      PadeGridGenerationTool *oneNewTool = new PadeGridGenerationTool(theFirstCoord,
								      theLength, 
								      theFirstResolution, 
								      theLastResolution,
								      theWaveLength);
      m_ToolVec.push_back(oneNewTool);
    }else{
      theFirstCoord  = currPnt.m_Coord;

      Standard_Integer theResolution = currPnt.m_Step; 
      theFirstResolution = theResolution;
      theLastResolution = theResolution;
      theLength = theBndBox.getUpperBound(m_Dir) - theFirstCoord + 0.25*theWaveLength/Standard_Real(theResolution);

      PadeGridGenerationTool *oneNewTool = new PadeGridGenerationTool(theFirstCoord,
								      theLength, 
								      theFirstResolution, 
								      theLastResolution,
								      theWaveLength);
      m_ToolVec.push_back(oneNewTool);
    }
  }
}


void 
PadeGrid_Tool::
Build()
{
  m_CoordinateVec.clear();

  vector<PadeGridGenerationTool*>::iterator iter;
  vector<PadeGridGenerationTool*>::iterator next_iter;

  for(iter=m_ToolVec.begin(); iter!=m_ToolVec.end(); iter++){
    (*iter)->Build();
  }

  for(iter=m_ToolVec.begin(); iter!=m_ToolVec.end(); iter++){
    next_iter = iter+1;
    PadeGridGenerationTool* currTool = *iter;
    const vector<Standard_Real>& currData = currTool->GetResult();

    if(!currData.empty()){
      Standard_Size nb = currData.size();
      for(Standard_Size i=0; i<(nb-1); i++){
	m_CoordinateVec.push_back(currData[i]);
      }
      if(next_iter==m_ToolVec.end()){
	m_CoordinateVec.push_back(currData[nb-1]);
      }
    }
  }
}


void 
PadeGrid_Tool::
ClearTools()
{
  vector<PadeGridGenerationTool*>::iterator iter;

  for(iter=m_ToolVec.begin(); iter!=m_ToolVec.end(); iter++){
    PadeGridGenerationTool* tmpPtr = *iter;
    *iter = NULL;
    delete tmpPtr;
  }
}
