#include <GridGeometry.hxx>
#include <PortDataFunc.hxx>

//#define PML_DBG


void GridGeometry::SetPMLAccordingPorts()
{
  if(GetPMLDefineTool() == NULL) return;

  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = this->GetGridBndDatas()->GetPorts();
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;

  for(iter = thePorts->begin(); iter!=thePorts->end(); iter++){
    const PortData& thePort = iter->second;

    Standard_Integer thePortDir = thePort.m_Dir;
    Standard_Integer thePortType = thePort.m_Type;

    if(!IsPMLPortType(thePort.m_Type)) continue;


    TxSlab2D<Standard_Integer> thePMLRgn;
    Standard_Integer theStartIndex;
    Standard_Integer thePMLLayerNum;
    
    ComputePMLInfAccordingOpenPort(thePort,thePMLRgn, theStartIndex,thePMLLayerNum);
    thePMLLayerNum = thePMLLayerNum-GetZRGrid()->GetMargin();
    
    SetPMLAccordingPorts_GridVertexDatas(thePortDir, theStartIndex, thePMLRgn, thePMLLayerNum);
    SetPMLAccordingPorts_GridEdgeDatas(thePortDir, theStartIndex, thePMLRgn, thePMLLayerNum);
    SetPMLAccordingPorts_GridFaceDatas(thePortDir, theStartIndex, thePMLRgn, thePMLLayerNum);
    
#ifdef PML_DBG
    cout<<"PortRgn---------------------------------------------------->>>"<<endl;
    thePMLRgn.write(cout);
    cout<<"theStartIndex = "<<theStartIndex<<endl;
    cout<<"thePMLLayerNum = "<<thePMLLayerNum<<endl;
    cout<<"PortRgn----------------------------------------------------<<<"<<endl;
#endif
  } // for(iter=......)
}



void 
GridGeometry::
SetPMLAccordingPorts_GridVertexDatas(const Standard_Integer thePortDir, 
				     const Standard_Integer theStartIndex,
				     const TxSlab2D<Standard_Integer>& thePMLRgn,
				     const Standard_Integer thePMLLayerNum)
{
  Standard_Real theStartLocation = GetZRGrid()->GetOrg()[thePortDir] + GetZRGrid()->GetLength(thePortDir, theStartIndex);
  Standard_Real theGridStep = GetZRGrid()->GetStep(thePortDir, theStartIndex);
      
  // 1.0 get the all GridEdgeDatas whose material are not PEC
  vector<GridVertexData*>  theVertexDatas;
  GetGridVertexDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),thePMLRgn, false, theVertexDatas);
  
  for(Standard_Size nv = 0; nv<theVertexDatas.size(); nv++){
    TxVector2D<Standard_Real> theVPnt = theVertexDatas[nv]->GetLocation();
    Standard_Real theRealDistance = theVPnt[thePortDir] - theStartLocation;
    
    Standard_Real theDistance = fabs(theRealDistance);
    
    Standard_Real theSigma,theAlpha, theKappa;
    if(GetPMLDefineTool()->GetMethodKey()==1){
      theSigma = GetPMLDefineTool()->ComputePMLSigma(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa(theGridStep, theDistance, thePMLLayerNum);
    }else{
      theSigma = GetPMLDefineTool()->ComputePMLSigma_2(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa_2(theGridStep, theDistance, thePMLLayerNum);
    }
    theAlpha = GetPMLDefineTool()->ComputePMLAlpha(theGridStep, theDistance, thePMLLayerNum);
    
    theVertexDatas[nv]->SetMaterialType( PML );
    theVertexDatas[nv]->SetupPMLData();
    
    theVertexDatas[nv]->SetPMLSigma(thePortDir, theSigma);
    theVertexDatas[nv]->SetPMLAlpha(thePortDir, theAlpha);
    theVertexDatas[nv]->SetPMLKappa(thePortDir, theKappa);
  }
}



void 
GridGeometry::
SetPMLAccordingPorts_GridEdgeDatas(const Standard_Integer thePortDir, 
				   const Standard_Integer theStartIndex,
				   const TxSlab2D<Standard_Integer>& thePMLRgn,
				   const Standard_Integer thePMLLayerNum)
{
  Standard_Real theStartLocation = GetZRGrid()->GetOrg()[thePortDir] + GetZRGrid()->GetLength(thePortDir, theStartIndex);
  Standard_Real theGridStep = GetZRGrid()->GetStep(thePortDir, theStartIndex);
      
  // 1.0 get the all GridEdgeDatas whose material are not PEC
  vector<GridEdgeData*>  theEdgeDatas;
  GetGridEdgeDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),thePMLRgn, false, theEdgeDatas);
  
  for(Standard_Size ne = 0; ne<theEdgeDatas.size(); ne++){
    TxVector2D<Standard_Real> theEdgeMidPnt;
    theEdgeDatas[ne]->ComputeMidPntLocation(theEdgeMidPnt);
    Standard_Real theRealDistance = theEdgeMidPnt[thePortDir] - theStartLocation;
    
    Standard_Real theDistance = fabs(theRealDistance);
    
    Standard_Real theSigma,theAlpha, theKappa;
    if(GetPMLDefineTool()->GetMethodKey()==1){
      theSigma = GetPMLDefineTool()->ComputePMLSigma(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa(theGridStep, theDistance, thePMLLayerNum);
    }else{
      theSigma = GetPMLDefineTool()->ComputePMLSigma_2(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa_2(theGridStep, theDistance, thePMLLayerNum);
    }
    theAlpha = GetPMLDefineTool()->ComputePMLAlpha(theGridStep, theDistance, thePMLLayerNum);
    
    theEdgeDatas[ne]->SetMaterialType( PML );
    theEdgeDatas[ne]->SetupPMLData();
    
    theEdgeDatas[ne]->SetPMLSigma(thePortDir, theSigma);
    theEdgeDatas[ne]->SetPMLAlpha(thePortDir, theAlpha);
    theEdgeDatas[ne]->SetPMLKappa(thePortDir, theKappa);
  }
}



void 
GridGeometry::
SetPMLAccordingPorts_GridFaceDatas(const Standard_Integer thePortDir, 
				   const Standard_Integer theStartIndex,
				   const TxSlab2D<Standard_Integer>& thePMLRgn,
				   const Standard_Integer thePMLLayerNum)
{
  Standard_Real theStartLocation = GetZRGrid()->GetOrg()[thePortDir] + GetZRGrid()->GetLength(thePortDir, theStartIndex);
  Standard_Real theGridStep = GetZRGrid()->GetStep(thePortDir, theStartIndex);

  vector<GridFaceData*> theFaceDatas;
  GetGridFaceDatasNotOfMaterialTypeOfSubRgn( Standard_Integer(PEC),thePMLRgn, theFaceDatas);
  
  for(Standard_Size nf = 0; nf<theFaceDatas.size(); nf++){
    TxVector2D<Standard_Real> theFaceBaryCenter = theFaceDatas[nf]->GetBaryCenter();
    Standard_Real theRealDistance = theFaceBaryCenter[thePortDir] - theStartLocation;
    Standard_Real theDistance = fabs(theRealDistance);
    
    Standard_Real theSigma,theAlpha, theKappa;
    
    if(GetPMLDefineTool()->GetMethodKey()==1){
      theSigma = GetPMLDefineTool()->ComputePMLSigma(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa(theGridStep, theDistance, thePMLLayerNum);
    }else{
      theSigma = GetPMLDefineTool()->ComputePMLSigma_2(theGridStep, theDistance, thePMLLayerNum);
      theKappa = GetPMLDefineTool()->ComputePMLKappa_2(theGridStep, theDistance, thePMLLayerNum);
    }
    
    theAlpha = GetPMLDefineTool()->ComputePMLAlpha(theGridStep, theDistance, thePMLLayerNum);
    
    theFaceDatas[nf]->SetMaterialType( PML );
    theFaceDatas[nf]->SetupPMLData();
    
    theFaceDatas[nf]->SetPMLSigma(thePortDir, theSigma);
    theFaceDatas[nf]->SetPMLAlpha(thePortDir, theAlpha);
    theFaceDatas[nf]->SetPMLKappa(thePortDir, theKappa);
  }
}
